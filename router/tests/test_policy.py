"""routing policy: decide() on hand-built backend states (no I/O)"""
import unittest

from tests import helpers  # noqa: F401
from tfrouter import config as cfgmod
from tfrouter.backend import Backend, Inflight
from tfrouter.policy import ReqCtx, decide


def setup(**ov):
    ov.setdefault("backends", {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2"}})
    cfg = cfgmod.load(None, ov)
    bs = [Backend(c, cfg) for c in cfg["backends"].values()]
    return cfg, bs, {b.name: b for b in bs}


def load(b, now, decoding=0, waiting_pf=(), out=300):
    """put `decoding` streaming requests and prefills of the given predicted seconds on a backend"""
    for i in range(decoding):
        r = Inflight("d%s%d" % (b.name, i), "interactive", True, 2000, 2000, 0, out)
        b.admit(r, now, 1.0, 0.5)
        b.first_token(r, now)
    for i, pf in enumerate(waiting_pf):
        r = Inflight("w%s%d" % (b.name, i), "interactive", True, 8000, 8000, 0, 100)
        b.admit(r, now, pf, pf)


def ctx(n_in, cls="interactive", out=300, cached=None, exact=True, **kw):
    return ReqCtx("t", cls, n_in, exact, out, out, cached or {}, **kw)


class PolicyTests(unittest.TestCase):
    def test_idle_chat_goes_to_primary(self):
        cfg, bs, _ = setup()
        d = decide(cfg, bs, ctx(2000), 0.0)
        self.assertEqual((d.choice, d.rule), ("S", "min-cost"))
        self.assertEqual(d.ranking, ["S", "M"])

    def test_overflow_to_secondary_when_primary_is_backed_up(self):
        cfg, bs, by = setup()
        load(by["S"], 0.0, decoding=4, waiting_pf=(12.0, 8.0))     # S: 20 s of prefill queued, all slots decoding
        d = decide(cfg, bs, ctx(2000), 0.0)
        self.assertEqual(d.choice, "M")

    def test_prefilling_burst_reserves_future_decode_capacity(self):
        cfg, bs, by = setup(
            backends={
                "S": {"url": "http://s:1", "prefill_pts": [[0, 0.01], [10000, 0.01]],
                      "decode": {"round_ms_pts": [[1, 10], [4, 60]],
                                 "tokens_per_round_pts": [[1, 1], [4, 1]]}},
                "M": {"url": "http://m:2", "prefill_pts": [[0, 0.01], [10000, 0.01]],
                      "decode": {"round_ms_pts": [[1, 20], [4, 40]],
                                 "tokens_per_round_pts": [[1, 1], [4, 1]]}},
            },
            policy={"primary_bias_s": 0.0, "primary_bias_frac": 0.0},
        )
        self.assertEqual(decide(cfg, bs, ctx(64, out=256), 0.0).choice, "S")
        for i in range(3):
            rec = Inflight("prefilling%d" % i, "interactive", True, 64, 64, 0, 256)
            by["S"].admit(rec, 0.0, 0.01, 0.01)
        self.assertEqual(by["S"].view(0.0).k, 0)
        self.assertEqual(decide(cfg, bs, ctx(64, out=256), 0.0).choice, "M")

    def test_prompt_above_secondary_context_goes_primary(self):
        cfg, bs, by = setup()
        load(by["S"], 0.0, waiting_pf=(60.0,))                       # even with S badly backed up
        d = decide(cfg, bs, ctx(50000), 0.0)
        self.assertEqual(d.choice, "S")
        self.assertEqual(d.cands["M"]["excluded"], "context")

    def test_sampled_grammar_and_image_requests_stay_off_the_greedy_only_secondary_with_pd_off(self):
        cfg, bs, by = setup()
        load(by["S"], 0.0, decoding=4, waiting_pf=(12.0, 8.0))     # M is the cheaper backend for a plain request
        self.assertEqual(decide(cfg, bs, ctx(2000), 0.0).choice, "M")
        for why in ("sampled", "field:guided_json", "images"):
            d = decide(cfg, bs, ctx(2000, s_only=why), 0.0)
            self.assertEqual((d.choice, d.cands["M"]["excluded"]), ("S", "s-only:" + why))
        d = decide(cfg, bs, ctx(2000, s_only="exact"), 0.0)          # `exact` only rules out the hand-off
        self.assertEqual(d.choice, "M")
        d = decide(cfg, bs, ctx(2000, s_only="sampled", force="M"), 0.0)   # an explicit pin is the operator's call
        self.assertEqual(d.choice, "M")

    def test_context_counts_output_and_estimates_get_a_margin(self):
        cfg, bs, _ = setup()
        d = decide(cfg, bs, ctx(40800, out=300), 0.0)                # 40800 + 300 > 40960
        self.assertEqual(d.cands["M"]["excluded"], "context")
        d = decide(cfg, bs, ctx(39500, out=300, exact=False), 0.0)    # fits only if the estimate is exact
        self.assertEqual(d.cands["M"]["excluded"], "context")
        d = decide(cfg, bs, ctx(39500, out=300, exact=True), 0.0)
        self.assertNotIn("excluded", d.cands["M"])

    def test_nothing_fits_is_413(self):
        cfg, bs, _ = setup()
        d = decide(cfg, bs, ctx(2_000_000), 0.0)
        self.assertEqual((d.status, d.choice), (413, None))

    def test_guard_keeps_long_prompt_off_a_decoding_secondary(self):
        cfg, bs, by = setup()
        load(by["S"], 0.0, decoding=4, waiting_pf=(30.0,))           # S is miserable
        load(by["M"], 0.0, decoding=2)                               # M has two streams decoding
        d = decide(cfg, bs, ctx(12000), 0.0)
        self.assertEqual(d.choice, "S")
        self.assertEqual(d.cands["M"]["guard"], "long-prompt-while-decoding")
        # the same prompt on an IDLE M is allowed
        cfg, bs, by = setup()
        load(by["S"], 0.0, decoding=4, waiting_pf=(30.0,))
        self.assertEqual(decide(cfg, bs, ctx(12000), 0.0).choice, "M")

    def test_guard_relaxed_when_it_would_leave_nothing(self):
        cfg, bs, by = setup()
        by["S"].breaker.force_open(0.0)
        load(by["M"], 0.0, decoding=2)
        d = decide(cfg, bs, ctx(12000), 0.0)
        self.assertEqual(d.choice, "M")
        self.assertTrue(d.rule.startswith("guard-relaxed"))

    def test_background_goes_to_secondary(self):
        cfg, bs, by = setup()
        d = decide(cfg, bs, ctx(12000, cls="background"), 0.0)       # both idle: M gets the batch work (bonus), S stays free for chat
        self.assertEqual(d.choice, "M")
        # unless M's queue is deeper than the cap
        load(by["M"], 0.0, waiting_pf=(100.0, 100.0, 100.0))
        d = decide(cfg, bs, ctx(12000, cls="background"), 0.0)
        self.assertEqual(d.choice, "S")

    def test_background_has_no_slo_interactive_does(self):
        cfg, bs, by = setup()
        load(by["S"], 0.0, waiting_pf=(40.0,))
        load(by["M"], 0.0, waiting_pf=(40.0,))
        di = decide(cfg, bs, ctx(2000), 0.0)
        self.assertIn("slo-miss", di.rule)
        db = decide(cfg, bs, ctx(2000, cls="background"), 0.0)
        self.assertNotIn("slo-miss", db.rule)

    def test_prefix_affinity_beats_idle_primary(self):
        cfg, bs, _ = setup()
        # 6k-token conversation, the secondary holds 5.7k of it: 300 rows on M (1.0 s) vs 6000 rows on S (3.3 s) -> M for a short answer ...
        d = decide(cfg, bs, ctx(6000, out=60, cached={"M": 5700}), 0.0)
        self.assertEqual(d.choice, "M")
        self.assertEqual(d.cands["M"]["rows"], 300)
        # ... but with a 300-token answer S's faster decode (58 vs 39 tok/s) cancels the prefill gain: the cost is E2E, not TTFT
        self.assertEqual(decide(cfg, bs, ctx(6000, out=300, cached={"M": 5700}), 0.0).choice, "S")
        # mode ll (TTFT only) always takes the cache hit
        cfg2, bs2, _ = setup(policy={"mode": "ll"})
        self.assertEqual(decide(cfg2, bs2, ctx(6000, out=300, cached={"M": 5700}), 0.0).choice, "M")
        # a cache hit on S only strengthens the default
        d = decide(cfg, bs, ctx(6000, cached={"S": 5700}), 0.0)
        self.assertEqual(d.choice, "S")

    def test_breaker_open_removes_a_backend(self):
        cfg, bs, by = setup()
        by["S"].breaker.force_open(0.0)
        d = decide(cfg, bs, ctx(2000), 1.0)
        self.assertEqual(d.choice, "M")
        self.assertEqual(d.cands["S"]["excluded"], "breaker-open")
        by["M"].breaker.force_open(0.0)
        d = decide(cfg, bs, ctx(2000), 1.0)
        self.assertEqual((d.status, d.rule), (503, "no-backend"))

    def test_forced_backend(self):
        cfg, bs, _ = setup()
        d = decide(cfg, bs, ctx(2000, force="M"), 0.0)
        self.assertEqual((d.choice, d.rule), ("M", "forced"))
        d = decide(cfg, bs, ctx(60000, force="M"), 0.0)                # forced but it does not fit
        self.assertEqual(d.status, 413)

    def test_ll_mode_ignores_interference(self):
        # S is slower to first token by 0.2 s only; M's two running streams would be stalled by an 8k prefill
        cfg, bs, by = setup(policy={"mode": "ll", "primary_bias_s": 0.0})
        load(by["S"], 0.0, waiting_pf=(5.0,))
        load(by["M"], 0.0)
        by["M"].guard_rows = None
        d = decide(cfg, bs, ctx(3000, out=50, slo_ttft=100.0), 0.0)
        self.assertEqual(d.rule, "min-ttft")
        cfg2, bs2, by2 = setup(policy={"mode": "cost", "primary_bias_s": 0.0})
        load(by2["S"], 0.0, waiting_pf=(5.0,))
        load(by2["M"], 0.0, decoding=3)
        by2["M"].guard_rows = None
        dc = decide(cfg2, bs2, ctx(3000, out=50), 0.0)
        self.assertGreater(dc.cands["M"]["pred"]["interf"], 0.0)

    def test_slo_override_and_primary_bias(self):
        cfg, bs, by = setup()
        load(by["S"], 0.0, waiting_pf=(2.0,))
        d = decide(cfg, bs, ctx(1000, slo_ttft=0.5), 0.0)              # nobody can make 0.5 s
        self.assertIn("slo-miss", d.rule)
        self.assertIn(d.choice, ("S", "M"))

    def test_reject_when_predicted_ttft_exceeds_limit(self):
        cfg, bs, by = setup(policy={"reject_ttft_s": 30.0})
        load(by["S"], 0.0, waiting_pf=(100.0,))
        load(by["M"], 0.0, waiting_pf=(100.0,))
        d = decide(cfg, bs, ctx(2000), 0.0)
        self.assertEqual((d.status, d.rule), (503, "overload"))


if __name__ == "__main__":
    unittest.main()
