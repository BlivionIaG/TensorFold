"""unit tests of the pure parts: config, cost model (incl. equivalence with bench/sim.py), prefix index, token estimator, breaker, backend bookkeeping"""
import json
import os
import random
import sys
import tempfile
import unittest

from tests import helpers  # noqa: F401  (sys.path)
from tfrouter import config as cfgmod
from tfrouter.backend import Backend, Breaker, Inflight, KeyFile
from tfrouter.model import Profile, View, interp, predict
from tfrouter.prefix import PrefixIndex
from tfrouter.tokens import TokenEstimator, request_size

try:
    import sim
except ImportError:                       # pragma: no cover
    sim = None


def cfg(**ov):
    ov.setdefault("backends", {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2"}})
    return cfgmod.load(None, ov)


class ConfigTests(unittest.TestCase):
    def test_defaults_valid_and_measured(self):
        c = cfg()
        s, m = c["backends"]["S"], c["backends"]["M"]
        self.assertEqual((s["slots"], m["slots"]), (4, 4))
        self.assertEqual(m["max_context"], 40960)
        self.assertEqual(s["role"], "primary")
        self.assertEqual(m["guard_rows"], 4096)
        self.assertEqual(dict(map(tuple, s["prefill_pts"]))[4055], 2.22)
        self.assertEqual(c["policy"]["mode"], "cost")

    def test_unknown_key_is_an_error(self):
        with self.assertRaises(ValueError):
            cfg(policy={"modee": "cost"})
        with self.assertRaises(ValueError):
            cfg(backends={"S": {"url": "http://s:1", "slotz": 3}})

    def test_backend_needs_url_and_preset(self):
        with self.assertRaises(ValueError):
            cfgmod.load(None, {})                         # S and M without urls
        with self.assertRaises(ValueError):
            cfg(backends={"X": {"url": "http://x:1"}})    # no preset
        c = cfg(backends={"X": {"url": "http://x:1", "preset": "M", "max_context": 1000}})
        self.assertEqual(c["backends"]["X"]["max_context"], 1000)

    def test_shipped_json_matches_defaults(self):
        path = os.path.join(helpers.ROOT, "router.config.json")
        with open(path) as f:
            shipped = json.load(f)
        self.assertEqual(shipped, cfgmod.dump_defaults(), "router.config.json is stale: python3 -m tfrouter --dump-config > router.config.json")

    def test_user_file_overrides(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"policy": {"w_int": 2.5, "classes": {"background": {"w_int": 3.0}}}, "backends": {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2", "slots": 8}}}, f)
        try:
            c = cfgmod.load(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(c["policy"]["w_int"], 2.5)
        self.assertEqual(c["policy"]["classes"]["background"]["w_int"], 3.0)
        self.assertEqual(c["policy"]["classes"]["background"]["prefer_role"], "secondary")   # merged, not replaced
        self.assertEqual(c["backends"]["M"]["slots"], 8)

    def test_config_files_merge_in_order_and_the_shipped_presets_are_valid_overrides(self):
        presets = os.path.join(helpers.ROOT, "presets", "current-engines.json")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"backends": {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2", "slots": 3}}}, f)
        try:
            c = cfgmod.load([presets, f.name])
        finally:
            os.unlink(f.name)
        self.assertEqual(c["backends"]["M"]["slots"], 3)                                  # the later file wins
        self.assertEqual(c["backends"]["S"]["decode"]["round_ms_pts"][0][0], 1)           # the shipped presets came through
        self.assertLess(c["backends"]["S"]["prefill_pts"][0][1], c["backends"]["M"]["prefill_pts"][0][1])


class ModelTests(unittest.TestCase):
    def setUp(self):
        c = cfg()
        self.S, self.M = Profile(c["backends"]["S"]), Profile(c["backends"]["M"])

    def test_interp(self):
        self.assertAlmostEqual(interp([(0, 1), (10, 3)], 5), 2.0)
        self.assertAlmostEqual(interp([(0, 1), (10, 3)], 20), 5.0)       # linear extrapolation
        self.assertEqual(interp([(5, 1), (10, 3)], 0), 1)

    def test_measured_tables(self):
        self.assertAlmostEqual(self.S.pf_seconds(4055)[0], 2.22)          # baselines.md s2
        self.assertAlmostEqual(self.M.pf_seconds(8280)[0], 8.74)
        self.assertAlmostEqual(self.S.stream_tps(1) * 1.0, 1 / (0.0308 / 1.79), places=6)
        self.assertAlmostEqual(4 / (self.S.step_s(4) / self.S.tpe(4) * 4) * 4, 4 * self.S.stream_tps(4), places=6)
        self.assertAlmostEqual(4 * self.S.stream_tps(4), 109.7, delta=0.5)   # S aggregate at 4 streams
        self.assertAlmostEqual(4 * self.M.stream_tps(4), 60.6, delta=0.3)    # M aggregate at 4 streams
        self.assertAlmostEqual(self.M.stream_tps(1), 39.4, delta=0.1)

    def test_prefill_interleave_inflation(self):
        p = predict(self.S, View(k=2), 8192, 100, 0.0)
        q = predict(self.S, View(k=0), 8192, 100, 0.0)
        self.assertAlmostEqual(p.infl, 4 * 0.25, places=6)               # 4 chunks x 250 ms window
        self.assertEqual(q.infl, 0.0)
        self.assertEqual(predict(self.M, View(k=3), 8192, 100, 0.0).infl, 0.0)   # M blocks instead

    def test_interference_grows_with_k_and_prefill(self):
        a = predict(self.M, View(k=2), 8000, 200, 1.0)
        b = predict(self.M, View(k=0), 8000, 200, 1.0)
        self.assertGreater(a.interf, 2 * 8.0)                              # >= k x stall
        self.assertEqual(b.interf, 0.0)
        self.assertAlmostEqual(b.cost, b.e2e)

    @unittest.skipIf(sim is None, "bench/sim.py not importable")
    def test_equals_sim_predict_on_live_states(self):
        """the router's predict() == sim.predict() (P == D) on machine states reached by a loaded simulation"""
        prm = sim.PARAM
        # tables the same as sim.PARAM (the presets were copied from it)
        for name, pre in (("S", "S_prod_pf"), ("M", "M_pf_pts")):
            prof = self.S if name == "S" else self.M
            for rows in (1, 39, 700, 4096, 9000, 30000, 100000):
                self.assertAlmostEqual(prof.pf_table(rows), sim.interp(prm[pre], rows) if name == "S" else sim.interp(prm[pre], rows), places=6)
        for k in (1, 2, 3, 4, 6):
            self.assertAlmostEqual(self.S.step_s(k), sim.step_S(prm, k), places=6)
            self.assertAlmostEqual(self.S.tpe(k), sim.tpe_S(prm, k), places=6)
            self.assertAlmostEqual(self.M.step_s(k), sim.step_M(prm, k), delta=1e-4)
        rng = random.Random(3)
        reqs, _ = sim.WORKLOADS["mixed"][0](rng, 0.14, 400.0)
        s = sim.Sim(prm, sim.SETUPS["A+B least-loaded"], reqs, [], 400.0, 1)
        n_checked = 0
        for t_stop in range(40, 400, 20):
            s.run(float(t_stop))
            for P in ("S", "M"):
                m = s.mach[P]
                prof = self.S if P == "S" else self.M
                for rows, n_out in ((900, 200), (4000, 400), (16000, 256), (30000, 256)):
                    r = sim.Req("x", s.t, rows, n_out, "chat")
                    want, ttft_w, e2e_w = sim.predict(s, r, P, P, 1.0)
                    v = View(k=len(m.active), slot_wait=m.slot_wait(), backlog=m.backlog())
                    got = predict(prof, v, rows, n_out, 1.0)
                    self.assertAlmostEqual(got.cost, want, places=6)
                    self.assertAlmostEqual(got.ttft, ttft_w, places=6)
                    self.assertAlmostEqual(got.e2e, e2e_w, places=6)
                    n_checked += 1
        self.assertGreater(n_checked, 50)


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.ix = PrefixIndex(cfg())

    def msgs(self, n, tag="a", chars=3000):
        out = [{"role": "system", "content": "sys"}]
        for i in range(n):
            out.append({"role": "user", "content": "%s%d" % (tag, i) + "u" * chars})
            out.append({"role": "assistant", "content": "r%d" % i + "x" * 400})
        return out

    def test_pool_churn_evicts_after_budget(self):
        ix = self.ix
        ix.budget = {"S": 100000}
        a = ix.anchors_chat(self.msgs(1)[:2])
        ix.record(a, "S", 20000, 0.0, uncached=20000)
        self.assertGreater(ix.lookup(a, "S", 1.0)[0], 0)
        other = ix.anchors_chat(self.msgs(1, tag="zz")[:2])
        ix.record(other, "S", 60000, 2.0, uncached=60000)             # 60k of churn after a: still inside the 100k pool
        self.assertGreater(ix.lookup(a, "S", 3.0)[0], 0)
        ix.record(other, "S", 60000, 4.0, uncached=60000)             # 120k > 100k: a is gone
        self.assertEqual(ix.lookup(a, "S", 5.0), (0, 0))
        self.assertGreater(ix.lookup(other, "S", 5.0)[0], 0)          # the entry that was touched last survives
        self.assertGreater(ix.lookup(a, "M", 5.0)[0] + 1, 0)          # (other backend has its own pool: unaffected, here simply unknown)

    def test_invalidate_drops_one_backend_only(self):
        ix = self.ix
        a = ix.anchors_chat(self.msgs(2))
        ix.record(a, "S", 9000, 0.0)
        ix.record(a, "M", 9000, 0.0)
        ix.invalidate(a, "S")
        self.assertEqual(ix.lookup(a, "S", 1.0), (0, 0))
        self.assertGreater(ix.lookup(a, "M", 1.0)[0], 0)

    def test_seed_separates_chains(self):
        ix = self.ix
        m = self.msgs(1)[:2]
        a1 = ix.anchors_chat(m, seed='[["tools"], null, null]')
        a2 = ix.anchors_chat(m, seed='[["other"], null, null]')
        a0 = ix.anchors_chat(m)
        ix.record(a1, "S", 9000, 0.0)
        self.assertGreater(ix.lookup(a1, "S", 1.0)[0], 0)
        self.assertEqual(ix.lookup(a2, "S", 1.0), (0, 0))
        self.assertEqual(ix.lookup(a0, "S", 1.0), (0, 0))

    def test_turn2_hits_turn1_backend(self):
        m1 = self.msgs(1)[:2]
        a1 = self.ix.anchors_chat(m1)
        self.ix.record(a1, "M", 1000, now=0.0)
        m2 = m1 + [{"role": "assistant", "content": "r0" + "x" * 400}, {"role": "user", "content": "next" + "q" * 500}]
        a2 = self.ix.anchors_chat(m2)
        tok, chars = self.ix.lookup(a2, "M", 1.0)
        self.assertEqual(tok, 1000)                      # the whole turn-1 prompt
        self.assertEqual(self.ix.lookup(a2, "S", 1.0), (0, 0))   # S never saw it

    def test_reply_anchor(self):
        m1 = self.msgs(1)[:2]
        a1 = self.ix.anchors_chat(m1)
        self.ix.record(a1, "S", 1000, 0.0)
        reply = "r0" + "x" * 400
        ext = self.ix.extend_with_reply(a1, reply)
        self.ix.record_extension(ext, a1.total_chars, "S", 1000, 120, 0.0)
        m2 = m1 + [{"role": "assistant", "content": reply}, {"role": "user", "content": "more"}]
        tok, _ = self.ix.lookup(self.ix.anchors_chat(m2), "S", 1.0)
        self.assertEqual(tok, 1120)                      # prompt + reply

    def test_shared_head_inside_one_message(self):
        doc = "d" * 50000
        a1 = self.ix.anchors_chat([{"role": "user", "content": doc + "question one?"}])
        self.ix.record(a1, "S", 15000, 0.0)
        a2 = self.ix.anchors_chat([{"role": "user", "content": doc + "a different question?"}])
        tok, chars = self.ix.lookup(a2, "S", 1.0)
        self.assertGreater(tok, 10000)                   # stride anchors inside the message
        self.assertLessEqual(chars, len(doc) + 10)

    def test_unrelated_prompt_misses(self):
        a1 = self.ix.anchors_chat(self.msgs(2, "a"))
        self.ix.record(a1, "S", 2000, 0.0)
        self.assertEqual(self.ix.lookup(self.ix.anchors_chat(self.msgs(2, "b")), "S", 1.0)[0], 0)

    def test_ttl_and_forget(self):
        a = self.ix.anchors_chat(self.msgs(1))
        self.ix.record(a, "S", 1000, 0.0)
        self.assertEqual(self.ix.lookup(a, "S", 100.0)[0], 1000)
        self.assertEqual(self.ix.lookup(a, "S", 30000.0)[0], 0)     # ttl 21600 s (safety net)
        self.ix.forget_backend("S")
        self.assertEqual(self.ix.lookup(a, "S", 1.0)[0], 0)

    def test_bounded(self):
        c = cfg(affinity={"max_entries": 50})
        ix = PrefixIndex(c)
        for i in range(40):
            ix.record(ix.anchors_chat(self.msgs(1, "t%d" % i)), "S", 1000, 0.0)
        self.assertLessEqual(len(ix.table), 50)

    def test_session(self):
        self.ix.record_session("conv-1", "M", 5000, 0.0)
        self.assertEqual(self.ix.lookup_session("conv-1", "M", 1.0, 6000), 5000)
        self.assertEqual(self.ix.lookup_session("conv-1", "M", 1.0, 3000), 2999)   # never the whole prompt
        self.assertEqual(self.ix.lookup_session("conv-1", "S", 1.0, 6000), 0)

    def test_min_prefix(self):
        a = self.ix.anchors_chat([{"role": "user", "content": "short"}])
        self.ix.record(a, "S", 100, 0.0)
        self.assertEqual(self.ix.lookup(a, "S", 1.0)[0], 0)           # below min_prefix_tokens


class TokenTests(unittest.TestCase):
    def test_estimate_and_calibration(self):
        t = TokenEstimator(cfg())
        chars, nm = request_size("chat", {"messages": [{"role": "user", "content": "x" * 3300}]})
        self.assertEqual((chars, nm), (3300, 1))
        e0 = t.estimate(chars, nm)
        for _ in range(60):                              # ground truth: 3.0 chars/token
            t.observe(3300, 1, 3300 // 3 + 4 + 2)
        self.assertAlmostEqual(t.cpt_of(), 3.0, delta=0.1)
        self.assertGreater(t.estimate(chars, nm), e0)    # more tokens than the 3.5 default thought

    def test_token_ids_are_exact(self):
        chars, nm = request_size("completions", {"prompt": [1, 2, 3, 4]})
        self.assertEqual(TokenEstimator(cfg()).estimate(chars, nm), 4)

    def test_boundary_detection(self):
        t = TokenEstimator(cfg())
        self.assertTrue(t.want_exact(40000, [40960, 4096]))
        self.assertFalse(t.want_exact(12000, [40960, 4096]))
        self.assertTrue(TokenEstimator(cfg(tokens={"mode": "exact"})).want_exact(10, []))
        self.assertFalse(TokenEstimator(cfg(tokens={"mode": "estimate"})).want_exact(40960, [40960]))

    def test_output_estimate(self):
        t = TokenEstimator(cfg())
        self.assertEqual(t.out_est("interactive", 100), 100)               # max_tokens below the EWMA
        self.assertEqual(t.out_est("interactive", 5000), 300)              # capped by the EWMA (default 300)
        self.assertEqual(t.out_est("interactive", 5000, hint=777), 777)
        for _ in range(100):
            t.observe_out("interactive", 600)
        self.assertGreater(t.out_est("interactive", 5000), 500)


class BreakerTests(unittest.TestCase):
    def test_cycle(self):
        b = Breaker(cfg()["breaker"])
        for t in range(3):
            self.assertTrue(b.allows(t))
            b.failure(t)
        self.assertEqual(b.state, "open")
        self.assertFalse(b.allows(5.0))
        self.assertTrue(b.allows(13.0))                   # cooldown 10 s -> half-open, one trial
        b.reserve(13.0)
        self.assertFalse(b.allows(13.1))                  # trial in flight
        b.failure(13.5)
        self.assertEqual(b.state, "open")
        self.assertEqual(b.cooldown, 20.0)                # doubled
        self.assertFalse(b.allows(25.0))
        self.assertTrue(b.allows(34.0))
        b.reserve(34.0)
        b.success(35.0)
        self.assertEqual((b.state, b.cooldown), ("closed", 10.0))

    def test_success_resets_failures(self):
        b = Breaker(cfg()["breaker"])
        b.failure(0)
        b.failure(1)
        b.success(2)
        b.failure(3)
        b.failure(4)
        self.assertEqual(b.state, "closed")


class KeyFileTests(unittest.TestCase):
    def test_formats_and_rotation(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("# comment\n\nlabel: sk-abc\n")
        try:
            k = KeyFile(f.name)
            self.assertEqual(k.get(), "sk-abc")
            os.utime(f.name, (1, 1))
            with open(f.name, "w") as g:
                g.write("admin[read]: sk-new\n")
            os.utime(f.name, None)
            k._checked = 0
            self.assertEqual(k.get(), "sk-new")
            with open(f.name, "w") as g:
                g.write("plainkey\n")
            os.utime(f.name, (2, 2))
            k._checked = 0
            self.assertEqual(k.get(), "plainkey")
        finally:
            os.unlink(f.name)


class BackendBookkeepingTests(unittest.TestCase):
    def mk(self, name="S"):
        c = cfg()
        return Backend(c["backends"][name], c)

    def rec(self, i, rows=4000, out=100, stream=True):
        return Inflight("r%d" % i, "interactive", stream, rows, rows, 0, out)

    def test_backlog_is_fifo_minus_head_progress(self):
        b = self.mk()
        r1, r2 = self.rec(1), self.rec(2)
        b.admit(r1, 0.0, 4.0, 4.0)
        b.admit(r2, 0.0, 6.0, 2.0)
        self.assertAlmostEqual(b.view(1.0).backlog, 6.0 - 1.0)
        b.first_token(r1, 4.0)                           # head done: r2 starts at 4.0
        self.assertAlmostEqual(b.view(5.0).backlog, 2.0 - 1.0)
        b.first_token(r2, 6.0)
        v = b.view(6.5)
        self.assertEqual((v.backlog, v.k), (0.0, 2))

    def test_slot_wait_and_queue(self):
        b = self.mk()
        recs = [self.rec(i, out=100) for i in range(5)]
        for r in recs:
            b.admit(r, 0.0, 1.0, 0.1)
        for r in recs[:4]:
            b.first_token(r, 1.0)
        v = b.view(2.0)
        self.assertEqual((v.k, v.queued, v.slots_used), (4, 1, 4))
        self.assertGreater(v.slot_wait, 0.0)             # all 4 slots are streaming: wait for the first to finish

    def test_non_stream_assumed_to_start_decoding_when_predicted(self):
        b = self.mk()
        r = self.rec(1, stream=False)
        b.admit(r, 0.0, 3.0, 3.0)
        self.assertEqual(b.view(1.0).k, 0)
        self.assertEqual(b.view(3.5).k, 1)

    def test_calibration_prefill_and_decode(self):
        b = self.mk("M")
        # a machine 1.5x slower than its table at prefill, 0.8x at decode
        for i in range(40):
            r = self.rec(i, rows=8000)
            t0 = i * 100.0
            b.admit(r, t0, 9.0, 9.0)
            b.first_token(r, t0 + 9.0 * 1.5)
            b.finish(r, t0 + 9.0 * 1.5 + 5.0, {"prompt_tokens": 8000, "completion_tokens": 100, "cached_tokens": 0, "prefill_seconds": b.profile.pf_table(8000) * 1.5,
                                               "decode_seconds": 99 / (39.4 * 0.8)})
        self.assertAlmostEqual(b.profile.pf_scale, 1.5, delta=0.1)
        self.assertAlmostEqual(b.profile.dec_scale, 0.8, delta=0.1)
        self.assertIsNotNone(b.ew_pf_tps.value)
        self.assertAlmostEqual(b.ew_dec_tps.value, 39.4 * 0.8, delta=2)

    def test_no_calibration_on_uncertain_cache(self):
        b = self.mk()
        r = self.rec(1, rows=8000)
        r.cached_est = 5000                                # we believed a prefix was cached, the backend does not say
        b.admit(r, 0.0, 1.0, 1.0)
        b.first_token(r, 0.5)
        b.finish(r, 2.0, {"prompt_tokens": 8000, "completion_tokens": 100, "cached_tokens": 0, "prefill_seconds": 0.5})
        self.assertEqual(b.profile.pf_scale, 1.0)

    def test_abort_releases(self):
        b = self.mk()
        r = self.rec(1)
        b.admit(r, 0.0, 4.0, 4.0)
        b.abort(r, 1.0, cancelled=True)
        v = b.view(1.0)
        self.assertEqual((len(b.inflight), v.backlog, b.counts["cancelled"]), (0, 0.0, 1))

    def test_external_load_needs_two_polls(self):
        b = self.mk()
        b.set_external(2)
        self.assertEqual(b.view(0).k, 0)
        b.set_external(2)
        self.assertEqual(b.view(0).k, 2)
        b.set_external(0)
        self.assertEqual(b.view(0).k, 0)


if __name__ == "__main__":
    unittest.main()
