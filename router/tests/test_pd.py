"""PD+ fleet: the SM path (S prefills with export, M pulls and decodes), the per-request prefill directive, S shadow bookkeeping (the wait queue / late binding is tests/test_admission.py)"""
import asyncio
import json
import unittest

from tests import helpers as h
from tests.test_policy import ctx, load, setup
from tfrouter import config as cfgmod
from tfrouter import pd
from tfrouter.backend import Inflight
from tfrouter.httpio import simple_request
from tfrouter.policy import decide
from tfrouter.server import Router
import fakepd

PD = {"pd": {"enabled": True}, "policy": {"primary_bias_frac": 0.0}}     # the SM thresholds of designs/pd-plus-fleet.md were set without the 15 % relative primary bias


def fast_m(**ov):
    """pd on and an M the router believes decodes 2.5x faster than today (the design's break-even region)"""
    cfg, bs, by = setup(**{**PD, **ov})
    by["M"].profile.dec_scale = 2.5
    return cfg, bs, by


class CostTests(unittest.TestCase):
    def test_handoff_matches_the_design(self):
        pdc = cfgmod.DEFAULTS["pd"]
        self.assertAlmostEqual(pd.handoff_s(pdc, 2048), 0.26, delta=0.01)             # design 3.1: 0.26 s at 2k, 0.27 at 16k, ~0.9-1.0 s at 64k
        self.assertAlmostEqual(pd.handoff_s(pdc, 16384), 0.27 + 0.05, delta=0.12)
        self.assertGreater(pd.handoff_s(pdc, 65536), 0.9)

    def test_predict_sm_is_s_prefill_plus_handoff_plus_m_decode(self):
        cfg, bs, by = setup(**PD)
        s, m = by["S"], by["M"]
        v = s.view(0.0)
        p = pd.predict_sm(cfg["pd"], s.profile, m.profile, v, m.view(0.0), 8000, 100, 1.0)
        tot, _ = s.profile.pf_seconds(8000)
        self.assertAlmostEqual(p.ttft, tot * 1.05 + pd.handoff_s(cfg["pd"], 8000), places=6)
        self.assertAlmostEqual(p.e2e, p.ttft + 99 * m.profile.tok_s(1), places=6)
        self.assertEqual(p.interf, 0.0)
        # S backlog delays it; running streams on S stall during the export, on M only during the landing
        load(s, 0.0, decoding=2, waiting_pf=(5.0,))
        p2 = pd.predict_sm(cfg["pd"], s.profile, m.profile, s.view(0.0), m.view(0.0), 8000, 100, 1.0)
        self.assertGreater(p2.ttft, p.ttft + 4.9)
        self.assertGreater(p2.interf, 2 * tot)


class PolicyTests(unittest.TestCase):
    def test_pd_off_has_no_sm_and_no_directive(self):
        cfg, bs, _ = setup()
        d = decide(cfg, bs, ctx(16000), 0.0)
        self.assertNotIn("SM", d.cands)
        self.assertEqual(sorted(d.paths), ["M", "S"])
        self.assertTrue(all(p.directive is None for p in d.paths.values()))

    def test_paths_carry_the_directives(self):
        cfg, bs, _ = fast_m()
        d = decide(cfg, bs, ctx(16000), 0.0)
        self.assertEqual((d.paths["S"].directive, d.paths["M"].directive, d.paths["SM"].directive), (None, "never", "force"))
        self.assertEqual((d.paths["SM"].backend, d.paths["SM"].prefill), ("M", "S"))
        self.assertEqual(d.paths["M"].prefill, None)

    def test_idle_with_todays_m_prefers_ss(self):
        cfg, bs, _ = setup(**PD)
        d = decide(cfg, bs, ctx(16000, out=512), 0.0)
        self.assertEqual(d.choice, "S")                      # SM = S + hand-off + M's slower decode: it cannot win idle TTFT or E2E
        self.assertIn("SM", d.ranking)

    def test_fast_m_idle_takes_sm_for_a_long_output(self):
        cfg, bs, _ = fast_m()
        d = decide(cfg, bs, ctx(16000, out=512), 0.0)
        self.assertEqual((d.choice, d.rule), ("SM", "min-cost"))
        self.assertEqual(d.preds["SM"].pf, d.preds["SM"].pf)           # a Pred is kept under the path name for the live bookkeeping
        self.assertGreater(d.cands["SM"]["handoff_s"], 0.2)

    def test_short_output_stays_on_s(self):
        cfg, bs, _ = fast_m()
        self.assertEqual(decide(cfg, bs, ctx(16000, out=8), 0.0).choice, "S")      # a hop that cannot pay for itself in few tokens

    def test_decoders_on_s_push_the_export_to_m_not_the_other_way(self):
        cfg, bs, by = fast_m()
        load(by["S"], 0.0, decoding=3)                       # three streams on S would stall for the whole export (SS and SM alike)
        d = decide(cfg, bs, ctx(16000, out=512), 0.0)
        self.assertEqual(d.choice, "M")
        self.assertGreater(d.cands["SM"]["pred"]["interf"], 0.0)

    def test_sampled_and_s_only_requests_exclude_m_and_sm(self):
        cfg, bs, _ = fast_m()
        d = decide(cfg, bs, ctx(16000, out=512, s_only="sampled"), 0.0)
        self.assertEqual((d.choice, d.ranking), ("S", ["S"]))
        self.assertEqual(d.cands["M"]["excluded"], "s-only:sampled")
        self.assertEqual(d.cands["SM"]["excluded"], "s-only:sampled")

    def test_exact_only_rules_out_the_handoff(self):
        cfg, bs, by = fast_m()
        d = decide(cfg, bs, ctx(16000, out=512, s_only="exact"), 0.0)
        self.assertEqual(d.cands["SM"]["excluded"], "s-only:exact")
        self.assertIn("_p", {**d.cands["M"], "_p": 1})                  # M stays a candidate
        self.assertNotIn("excluded", d.cands["M"])

    def test_sm_exclusions(self):
        cfg, bs, by = fast_m()
        self.assertEqual(decide(cfg, bs, ctx(3000, out=512), 0.0).cands["SM"]["excluded"], "too-few-rows")                  # below M's TF_PREFILL_MIN
        self.assertEqual(decide(cfg, bs, ctx(41000, out=512), 0.0).cands["SM"]["excluded"], "context")                       # above M's context
        d = decide(cfg, bs, ctx(16000, out=512, sm_off=True), 0.0)
        self.assertEqual(d.cands["SM"]["excluded"], "sm-fallbacks")
        by["S"].breaker.force_open(0.0)
        self.assertEqual(decide(cfg, bs, ctx(16000, out=512), 0.0).cands["SM"]["excluded"], "breaker-open-prefill")
        cfg, bs, by = fast_m()
        by["M"].breaker.force_open(0.0)
        d = decide(cfg, bs, ctx(16000, out=512), 0.0)
        self.assertEqual((d.choice, d.cands["SM"]["excluded"]), ("S", "breaker-open"))

    def test_rows_held_by_m_count_against_the_minimum(self):
        cfg, bs, _ = fast_m()
        d = decide(cfg, bs, ctx(16000, out=512, cached={"M": 14000}), 0.0)       # 2000 uncached rows on M: M finishes it itself
        self.assertEqual(d.cands["SM"]["excluded"], "too-few-rows")

    def test_forcing_a_path_by_name(self):
        cfg, bs, _ = fast_m()
        d = decide(cfg, bs, ctx(16000, out=512, force="SM"), 0.0)
        self.assertEqual((d.choice, d.rule, d.ranking), ("SM", "forced", ["SM"]))
        d = decide(cfg, bs, ctx(16000, out=512, force="S"), 0.0)
        self.assertEqual((d.choice, d.ranking), ("S", ["S"]))
        self.assertEqual(d.cands["SM"]["excluded"], "not-forced")

    def test_mu_charges_s_seconds_to_ss_and_sm_not_mm(self):
        cfg, bs, _ = fast_m()
        d = decide(cfg, bs, ctx(12000, out=300), 0.0)
        self.assertEqual(pd.extra_cost(cfg, d, "M", d.preds["M"]), 0.0)
        self.assertAlmostEqual(pd.extra_cost(cfg, d, "S", d.preds["S"]), d.preds["S"].pf)
        self.assertAlmostEqual(pd.extra_cost(cfg, d, "SM", d.preds["SM"]), d.preds["SM"].pf)
        cfg, bs, _ = fast_m(pd={"enabled": True, "mu": 0.0})
        d = decide(cfg, bs, ctx(12000, out=300), 0.0)
        self.assertEqual(pd.extra_cost(cfg, d, "S", d.preds["S"]), 0.0)

    def test_nothing_fits_is_still_413_with_pd_on(self):
        cfg, bs, _ = fast_m()
        d = decide(cfg, bs, ctx(2_000_000), 0.0)
        self.assertEqual(d.status, 413)


class FeatureTests(unittest.TestCase):
    def cfg(self, **pdov):
        return cfgmod.load(None, {"pd": dict({"enabled": True}, **pdov), "backends": {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2"}}})

    def test_features(self):
        c = self.cfg()
        self.assertIsNone(pd.features(c, {"messages": [{"role": "user", "content": "x"}]}))
        self.assertEqual(pd.features(c, {"temperature": 0.7}), "sampled")
        self.assertIsNone(pd.features(c, {"temperature": 0}))
        self.assertEqual(pd.features(c, {"response_format": {"type": "json_object"}}), "field:response_format")
        self.assertEqual(pd.features(c, {"logprobs": True}), "field:logprobs")
        self.assertEqual(pd.features(c, {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "x"}}]}]}), "images")
        self.assertEqual(pd.features(c, {}, " Exact "), "exact")
        self.assertIsNone(pd.features(c, {}, "any"))
        self.assertIsNone(pd.features(self.cfg(greedy_only_decode=False), {"temperature": 1.0}))

    def test_capability_rules_apply_with_the_hand_off_off(self):
        two = {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2"}}
        self.assertEqual(pd.features(cfgmod.load(None, {"backends": two}), {"temperature": 1.0}), "sampled")
        c = cfgmod.load(None, {"backends": two, "pd": {"greedy_only_decode": False}})
        self.assertIsNone(pd.features(c, {"temperature": 1.0}))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            cfgmod.load(None, {"pd": {"enabled": True, "decode_backend": "X"}, "backends": {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2"}}})
        with self.assertRaises(ValueError):
            cfgmod.load(None, {"pd": {"enabled": True, "decode_backend": "S"}, "backends": {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2"}}})


class BackendTests(unittest.TestCase):
    def test_shadow_prefill_queues_on_s_and_leaves_with_the_request(self):
        cfg, bs, by = setup(**PD)
        s, m = by["S"], by["M"]
        rec = Inflight("r1", "interactive", True, 16000, 16000, 0, 300)
        router = type("R", (), {"backends": by, "pd": pd.State(cfg["pd"]), "cfg": cfg})()
        d = decide(cfg, bs, ctx(16000, out=300, force="SM"), 0.0)
        pd.admit(router, rec, "SM", d, 0.0, d.preds["SM"])
        self.assertEqual((len(s.fifo), len(s.shadows), len(s.inflight)), (1, 1, 0))     # a prefill on S, no decode lane there
        self.assertEqual((len(m.fifo), len(m.pending), len(m.inflight)), (0, 1, 1))     # a lane on M, not in M's prefill queue
        self.assertGreater(s.view(0.0).backlog, 5.0)
        self.assertEqual(s.view(0.0).slots_used, 0)
        self.assertEqual(m.view(0.0).slots_used, 1)
        m.first_token(rec, 9.0)                                                         # M's first token: S's prefill is over
        self.assertEqual((len(s.fifo), len(s.shadows), len(m.pending)), (0, 0, 0))
        self.assertEqual(m.view(9.0).k, 1)

    def test_abort_drops_the_shadow_and_wakes_the_hold_queue(self):
        cfg, bs, by = setup(**PD)
        woke = []
        by["M"].on_free = lambda: woke.append(1)
        rec = Inflight("r1", "interactive", False, 16000, 16000, 0, 300)
        router = type("R", (), {"backends": by, "pd": pd.State(cfg["pd"]), "cfg": cfg})()
        d = decide(cfg, bs, ctx(16000, out=300, force="SM"), 0.0)
        pd.admit(router, rec, "SM", d, 0.0, d.preds["SM"])
        by["M"].abort(rec, 1.0, failed=True)
        self.assertEqual((len(by["S"].fifo), len(by["S"].shadows), len(by["M"].pending), len(by["M"].inflight)), (0, 0, 0, 0))
        self.assertEqual(woke, [1])

    def test_non_stream_sm_request_is_assumed_decoding_when_predicted(self):
        cfg, bs, by = setup(**PD)
        rec = Inflight("r1", "interactive", False, 16000, 16000, 0, 300)
        router = type("R", (), {"backends": by, "pd": pd.State(cfg["pd"]), "cfg": cfg})()
        d = decide(cfg, bs, ctx(16000, out=300, force="SM"), 0.0)
        pd.admit(router, rec, "SM", d, 0.0, d.preds["SM"])
        self.assertEqual(by["M"].view(1.0).k, 0)
        self.assertEqual(by["M"].view(d.preds["SM"].ttft + 1.0).k, 1)
        self.assertEqual(len(by["S"].shadows), 0)

    def test_external_load_does_not_count_our_exports(self):
        cfg, bs, by = setup(**PD)
        s = by["S"]
        rec = Inflight("r1", "interactive", True, 16000, 16000, 0, 300)
        router = type("R", (), {"backends": by, "pd": pd.State(cfg["pd"]), "cfg": cfg})()
        d = decide(cfg, bs, ctx(16000, out=300, force="SM"), 0.0)
        pd.admit(router, rec, "SM", d, 0.0, d.preds["SM"])
        for _ in range(3):                                   # S's server reports the export (M's 1-token request) as an active request
            s.set_external(1)
        self.assertEqual(s.view(0.1).k, 0)
        s.set_external(2)                                    # a second, foreign request is counted
        s.set_external(2)
        self.assertEqual(s.view(0.1).k, 1)

    def test_prefill_load_of_other_clients_adds_backlog(self):
        cfg, bs, by = setup(**PD)
        s = by["S"]
        base = s.view(0.0).backlog
        s.set_prefill_load(8192)
        self.assertGreater(s.view(0.0).backlog, base + 3.0)                              # 8k rows at ~1.8k rows/s
        rec = Inflight("r1", "interactive", True, 8192, 8192, 0, 100)
        s.admit(rec, 0.0, 5.0, 4.5)
        s.set_prefill_load(8192)                                                         # the ledger counts our own request: no double count
        self.assertEqual(s.ext_pf_tokens, 0)
        s.set_prefill_load(None)
        self.assertEqual(s.ext_pf_tokens, 0)

    def test_sm_does_not_calibrate_m_prefill(self):
        cfg, bs, by = setup(**PD)
        m = by["M"]
        rec = Inflight("r1", "interactive", True, 16000, 16000, 0, 300)
        router = type("R", (), {"backends": by, "pd": pd.State(cfg["pd"]), "cfg": cfg})()
        d = decide(cfg, bs, ctx(16000, out=300, force="SM"), 0.0)
        pd.admit(router, rec, "SM", d, 0.0, d.preds["SM"])
        m.first_token(rec, 9.0)
        m.finish(rec, 20.0, {"prompt_tokens": 16000, "completion_tokens": 100, "cached_tokens": 0})
        self.assertEqual(m.profile.pf_scale, 1.0)


class StateTests(unittest.TestCase):
    def test_fallbacks_disable_sm_for_a_while(self):
        st = pd.State(cfgmod.DEFAULTS["pd"])
        self.assertTrue(st.sm_outcome({"pulled_prefix": 5000}, 0.0))
        for i in range(2):
            self.assertFalse(st.sm_outcome({}, 1.0 + i))
        self.assertEqual(st.off_until, 0.0)
        self.assertTrue(st.sm_outcome({"remote_prefix": 5000}, 3.0))                      # a pulled one resets the run
        for i in range(3):
            self.assertFalse(st.sm_outcome({}, 10.0 + i))
        self.assertGreater(st.off_until, 60.0)
        self.assertEqual(st.counts["sm_disabled"], 1)


# ---------------------------------------------------------------------------------------------------------------------------
# the live proxy against fake machines: S = tools/fakebackend.py S, M = tools/fakepd.py (pulls from the S fake)
# ---------------------------------------------------------------------------------------------------------------------------
async def start_pair(speed=h.SPEED, m_decode_scale=2.5, s_kw=None, m_kw=None, ov=None, believe_fast=True):
    s = await h.start_fake("S", speed, **(s_kw or {}))
    a = fakepd.fakebackend.parse_args(["--profile", "M", "--speed", str(speed), "--decode-scale", str(m_decode_scale)])
    for k, v in (m_kw or {}).items():
        setattr(a, k, v)
    srv = fakepd.PdServer(a, ("127.0.0.1", s.port), 4096)
    srv.m.task = asyncio.ensure_future(srv.m.run())
    lis = await asyncio.start_server(srv.handle, "127.0.0.1", 0, limit=1 << 20)
    m = h.Fake(srv, lis, lis.sockets[0].getsockname()[1])
    m.pd = srv
    over = {"pd": {"enabled": True}, "policy": {"primary_bias_frac": 0.0}}
    h._deep(over, ov or {})
    r = await h.start_router({"S": s, "M": m}, over, speed)
    if believe_fast:
        r.backends["M"].profile.dec_scale = m_decode_scale
    return r, s, m


class ProxyTests(unittest.TestCase):
    def test_sm_end_to_end(self):
        async def go():
            r, s, m = await start_pair()
            try:
                st, hd, evs = await h.stream(r.port, h.chat_body(16000, stream=True, max_tokens=512, uniq="sm1"), [("X-Router-Explain", "1")])
                self.assertEqual(st, 200)
                dec = json.loads(hd["x-router-decision"])
                self.assertEqual(dec["choice"], "SM")
                self.assertEqual(hd["x-router-backend"], "M")                      # the request went to the M server ...
                self.assertEqual(m.pd.pdstats["force"], 1)                         # ... with the force directive ...
                self.assertEqual(m.pd.pdstats["pulled"], 1)
                self.assertEqual(s.server.m.stats["requests"], 1)                  # ... and M asked S for the export (the fake pair's trigger)
                usage = [json.loads(p) for _, p in evs if p != b"[DONE]" and b"tensorfold" in p][-1]
                self.assertGreater(usage["tensorfold"]["pulled_prefix"], 15000)
                await asyncio.sleep(0.05)
                self.assertEqual((r.backends["S"].fifo, r.backends["S"].shadows, r.backends["M"].pending), ([], [], []))
                self.assertEqual(r.pd.counts["SM"], 1)
                self.assertEqual(r.pd.counts["sm_pulled"], 1)
                st2, _, body = await h.http(r.port, "GET", "/router/state")
                self.assertEqual(json.loads(body)["pd"]["counts"]["SM"], 1)
            finally:
                await r.stop(), await s.close(), await m.close()
        h.run(go())

    def test_mm_carries_never_and_ss_carries_nothing(self):
        async def go():
            r, s, m = await start_pair()
            try:
                await h.http(r.port, "POST", "/v1/chat/completions", h.chat_body(600, uniq="mm", max_tokens=8), [("X-Route-To", "M")])
                self.assertEqual((m.pd.pdstats["never"], m.pd.pdstats["force"]), (1, 0))
                await h.http(r.port, "POST", "/v1/chat/completions", h.chat_body(600, uniq="ss", max_tokens=8), [("X-Route-To", "S")])
                self.assertEqual(m.pd.pdstats["never"], 1)
                self.assertEqual(s.server.m.stats["requests"], 1)
            finally:
                await r.stop(), await s.close(), await m.close()
        h.run(go())

    def test_sampled_request_never_reaches_m(self):
        async def go():
            r, s, m = await start_pair()
            try:
                body = h.chat_body(16000, uniq="t07", max_tokens=32, extra={"temperature": 0.7})
                st, hd, _ = await h.http(r.port, "POST", "/v1/chat/completions", body)
                self.assertEqual((st, hd["x-router-backend"]), (200, "S"))
                self.assertEqual(sum(m.pd.pdstats.values()), 0)
            finally:
                await r.stop(), await s.close(), await m.close()
        h.run(go())

    def test_route_header_names_sm(self):
        async def go():
            r, s, m = await start_pair(believe_fast=False)            # the router would pick S; the header overrides
            try:
                st, hd, _ = await h.http(r.port, "POST", "/v1/chat/completions", h.chat_body(8000, uniq="hdr", max_tokens=8), [("X-Route-To", "SM")])
                self.assertEqual((st, hd["x-router-backend"]), (200, "M"))
                self.assertEqual(m.pd.pdstats["pulled"], 1)
                st, _, _ = await h.http(r.port, "POST", "/v1/chat/completions", h.chat_body(800, uniq="bad", max_tokens=8), [("X-Route-To", "XX")])
                self.assertEqual(st, 400)
            finally:
                await r.stop(), await s.close(), await m.close()
        h.run(go())

    def test_route_header_sm_is_unknown_when_pd_is_off(self):
        async def go():
            r, s, m = await start_pair(ov={"pd": {"enabled": False}})
            try:
                st, _, _ = await h.http(r.port, "POST", "/v1/chat/completions", h.chat_body(800, uniq="off", max_tokens=8), [("X-Route-To", "SM")])
                self.assertEqual(st, 400)
            finally:
                await r.stop(), await s.close(), await m.close()
        h.run(go())

    def test_forced_requests_that_m_prefills_itself_switch_sm_off(self):
        async def go():
            r, s, m = await start_pair(ov={"pd": {"fallback": {"failures": 2, "cooldown_s": 600.0}}})
            m.pd.pair_down = True
            try:
                for i in range(2):
                    st, _, _ = await h.http(r.port, "POST", "/v1/chat/completions", h.chat_body(8000, uniq="fb%d" % i, max_tokens=8), [("X-Route-To", "SM")])
                    self.assertEqual(st, 200)
                self.assertEqual(r.pd.counts["sm_fallback"], 2)
                self.assertGreater(r.pd.off_until, r.clock.now())
                st, hd, _ = await h.http(r.port, "POST", "/v1/chat/completions", h.chat_body(16000, uniq="fb9", max_tokens=8), [("X-Router-Explain", "1")])
                dec = json.loads(hd["x-router-decision"])
                self.assertEqual(dec["candidates"]["SM"]["excluded"], "sm-fallbacks")
                self.assertNotEqual(dec["choice"], "SM")
            finally:
                await r.stop(), await s.close(), await m.close()
        h.run(go())


if __name__ == "__main__":
    unittest.main()
