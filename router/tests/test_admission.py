"""admission control (per-backend in-flight cap, router queue, 503 + Retry-After), the E2 low-load fixes (output-length warm-up, interference bounded by the
runners' remaining tokens, relative primary bias) and the regression tests built from the E2 run's 0.09 req/s split"""
import asyncio
import json
import unittest

from tests import helpers as h
from tests.test_policy import ctx, load, setup
from tfrouter import admission
from tfrouter import config as cfgmod
from tfrouter.backend import Backend, Inflight
from tfrouter.model import View, predict
from tfrouter.policy import decide
from tfrouter.tokens import TokenEstimator


# ---------------------------------------------------------------------------------------------------------------------------
# the wait queue and the lane predicates (no I/O)
# ---------------------------------------------------------------------------------------------------------------------------
class WaitQueueTests(unittest.TestCase):
    def test_kick_wakes_in_priority_then_age_order(self):
        async def go():
            q = admission.WaitQueue()
            order = []
            ws = [q.add(1), q.add(0), q.add(1), q.add(0)]       # background, interactive, background, interactive

            async def waiter(i, w):
                await q.wait(w, 5.0)
                order.append(i)
            tasks = [asyncio.ensure_future(waiter(i, w)) for i, w in enumerate(ws)]
            await asyncio.sleep(0)
            q.kick()
            await asyncio.gather(*tasks)
            return order
        self.assertEqual(h.run(go()), [1, 3, 0, 2])

    def test_wait_times_out(self):
        async def go():
            q = admission.WaitQueue()
            w = q.add(0)
            self.assertFalse(await q.wait(w, 0.01))
            q.remove(w)
            self.assertEqual(len(q), 0)
        h.run(go())

    def test_an_older_waiter_blocks_only_the_lanes_it_could_use(self):
        async def go():
            q = admission.WaitQueue()
            a = q.add(0, {"S"})                                                     # interactive, wants S
            self.assertTrue(q.blocked(0, admission.INF, {"S"}))                    # a newcomer may not take S's lane in front of it ...
            self.assertFalse(q.blocked(0, admission.INF, {"M"}))                   # ... but M's lane is nobody's
            self.assertTrue(q.blocked(1, admission.INF, {"S"}))                    # background never overtakes interactive
            self.assertFalse(q.blocked(*a[:2], {"S"}))                             # a waiter is not blocked by itself
            b = q.add(0, {"S", "M"})
            self.assertTrue(q.blocked(*b[:2], {"S"}))                              # the younger one waits behind the older
            self.assertFalse(q.blocked(*a[:2], {"S", "M"}))
            self.assertEqual(q.depth("S"), 2)
            self.assertEqual(q.depth("M"), 1)
        h.run(go())


class LaneTests(unittest.TestCase):
    def test_lane_free_counts_inflight_and_other_clients(self):
        cfg, bs, by = setup()
        s = by["S"]
        self.assertEqual((s.max_inflight, by["M"].max_inflight), (4, 4))            # default: the slot count
        for i in range(3):
            s.admit(Inflight("a%d" % i, "interactive", True, 2000, 2000, 0, 300), 0.0, 1.0, 1.0)
        self.assertTrue(s.lane_free())
        s.set_external(4)                                                           # other clients: 4 requests at the server, ours 3 -> 1
        s.set_external(4)                                                           # (two readings in a row make it real)
        self.assertEqual(s.in_use(), 4)
        self.assertFalse(s.lane_free())

    def test_waiters_ahead_lengthen_the_predicted_slot_wait(self):
        """the router's own queue is part of the price of waiting for a backend: the n-th waiter gets the n-th lane to free, then the lanes turn over again"""
        cfg, bs, by = setup()
        s = by["S"]
        recs = [Inflight("a%d" % i, "interactive", True, 2000, 2000, 0, 300) for i in range(4)]
        for i, rec in enumerate(recs):
            s.admit(rec, 0.0, 1.0, 1.0)
            s.first_token(rec, 0.0 + i)                                             # they finish one after the other
        waits = [s.view(5.0, ahead).slot_wait for ahead in (0, 1, 3, 4, 8)]
        self.assertEqual(waits, sorted(waits))
        self.assertEqual(len(set(waits)), len(waits))
        self.assertGreater(waits[-1], waits[2] * 1.5)                               # beyond the 4 running ones: another round of lanes, not the last runner's end again
        self.assertEqual(s.view(5.0).queued, 0)
        self.assertEqual(s.view(5.0, 3).queued, 3)

    def test_wait_queue_counts_the_waiters_ahead_per_backend(self):
        async def go():
            q = admission.WaitQueue()
            q.add(0, {"S", "M"}, {"S"})
            q.add(1, {"S", "M"}, {"M"})
            q.add(0, {"S"}, {"S"})
            self.assertEqual(q.ahead(0, admission.INF), {"S": 2})
            self.assertEqual(q.ahead(1, admission.INF), {"S": 2, "M": 1})
        h.run(go())

    def test_max_inflight_override_and_validation(self):
        cfg, bs, by = setup(backends={"S": {"url": "http://s:1", "max_inflight": 2}, "M": {"url": "http://m:2"}})
        self.assertEqual(by["S"].max_inflight, 2)
        with self.assertRaises(ValueError):
            setup(backends={"S": {"url": "http://s:1", "max_inflight": 0}, "M": {"url": "http://m:2"}})
        with self.assertRaises(ValueError):
            setup(admission={"max_wait_s": 0})


# ---------------------------------------------------------------------------------------------------------------------------
# the live proxy: caps, queue, 503s.  Both fakes have ONE slot, so two requests fill the fleet.
# ---------------------------------------------------------------------------------------------------------------------------
async def pair(ov=None, slots=1):
    s = await h.start_fake("S", slots=slots)
    m = await h.start_fake("M", slots=slots)
    over = {"backends": {"S": {"slots": slots}, "M": {"slots": slots}}}
    h._deep(over, ov or {})
    r = await h.start_router({"S": s, "M": m}, over)
    return r, s, m


async def close(r, s, m):
    await r.stop(), await s.close(), await m.close()


def long_body(tag, n=2000, tokens=3000):
    return h.chat_body(n, uniq=tag, stream=True, max_tokens=tokens)


class ProxyTests(unittest.TestCase):
    def test_in_flight_is_capped_at_the_slots_and_the_rest_waits_in_the_router(self):
        async def go():
            r, s, m = await pair({"admission": {"max_wait_s": 600.0}})
            try:
                tasks = []
                for i in range(6):
                    tasks.append(asyncio.ensure_future(h.stream(r.port, h.chat_body(2000, uniq="c%d" % i, stream=True, max_tokens=2000 if i < 2 else 5), [("X-Router-Explain", "1")])))
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.02)
                self.assertEqual(len(r.gate), 4)                                      # two lanes, six requests: four wait here
                self.assertEqual([b.in_use() for b in r.order], [1, 1])
                res = await asyncio.gather(*tasks)
                self.assertTrue(all(st == 200 for st, _, _ in res))
                for f in (s, m):
                    self.assertEqual(f.m.stats["max_queue"], 0)                       # nothing ever queued at a backend
                    self.assertLessEqual(f.m.stats["max_active"], 1)
                self.assertEqual(r.stats["held"], 4)
                held = sorted(json.loads(hd["x-router-decision"])["held_s"] > 0 for _, hd, _ in res)
                self.assertEqual(held, [False, False, True, True, True, True])
                self.assertEqual(len(r.gate), 0)
            finally:
                await close(r, s, m)
        h.run(go())

    def test_without_admission_the_backends_queue_the_excess(self):
        async def go():
            r, s, m = await pair({"admission": {"enabled": False}})
            try:
                tasks = [asyncio.ensure_future(h.stream(r.port, h.chat_body(2000, uniq="n%d" % i, stream=True, max_tokens=300))) for i in range(6)]
                res = await asyncio.gather(*tasks)
                self.assertTrue(all(st == 200 for st, _, _ in res))
                self.assertGreater(s.m.stats["max_queue"] + m.m.stats["max_queue"], 0)    # what E2 did to production
            finally:
                await close(r, s, m)
        h.run(go())

    def test_a_waiting_request_binds_to_whichever_backend_frees_first(self):
        async def go():
            r, s, m = await pair({"admission": {"max_wait_s": 600.0}})
            try:
                short = asyncio.ensure_future(h.stream(r.port, h.chat_body(2000, uniq="a", stream=True, max_tokens=800), [("X-Route-To", "M"), ("X-Expected-Output-Tokens", "800")]))
                long = asyncio.ensure_future(h.stream(r.port, h.chat_body(2000, uniq="b", stream=True, max_tokens=4000), [("X-Route-To", "S"), ("X-Expected-Output-Tokens", "4000")]))
                await asyncio.sleep(0.03)
                third = asyncio.ensure_future(h.stream(r.port, h.chat_body(2000, uniq="c", stream=True, max_tokens=20), [("X-Router-Explain", "1")]))
                st, hd, _ = await third
                self.assertEqual((st, hd["x-router-backend"]), (200, "M"))            # M's short request ended first; S was still busy
                self.assertGreater(json.loads(hd["x-router-decision"])["held_s"], 0)
                await asyncio.gather(short, long)
            finally:
                await close(r, s, m)
        h.run(go())

    def test_interactive_waiters_go_before_background_ones(self):
        async def go():
            r, s, m = await pair({"admission": {"max_wait_s": 600.0}})
            try:
                busy = [asyncio.ensure_future(h.stream(r.port, long_body("p%d" % i, tokens=3000))) for i in range(2)]
                await asyncio.sleep(0.05)
                first = {}

                async def one(tag, hdr):
                    st, hd, evs = await h.stream(r.port, h.chat_body(2000, uniq=tag, stream=True, max_tokens=10), hdr)
                    first[tag] = r.clock.now()
                    return st
                bg = asyncio.ensure_future(one("bg", [("X-Request-Class", "background")]))     # arrives first ...
                await asyncio.sleep(0.02)
                ia = asyncio.ensure_future(one("ia", []))                                       # ... interactive arrives second
                self.assertEqual(await asyncio.gather(bg, ia), [200, 200])
                self.assertLess(first["ia"], first["bg"])
                await asyncio.gather(*busy)
            finally:
                await close(r, s, m)
        h.run(go())

    def test_queue_full_is_503_with_retry_after(self):
        async def go():
            r, s, m = await pair({"admission": {"max_wait_s": 600.0}, "backends": {"S": {"queue_max": 1}, "M": {"queue_max": 1}}})
            try:
                busy = [asyncio.ensure_future(h.stream(r.port, long_body("q%d" % i))) for i in range(2)]
                await asyncio.sleep(0.05)
                waiter = asyncio.ensure_future(h.stream(r.port, h.chat_body(2000, uniq="w", stream=True, max_tokens=5)))
                await asyncio.sleep(0.05)
                self.assertEqual(len(r.gate), 1)
                st, hd, evs = await h.stream(r.port, h.chat_body(2000, uniq="x", stream=True, max_tokens=5))
                self.assertEqual(st, 503)                                              # both backends have their one waiter slot taken
                self.assertGreaterEqual(int(hd["retry-after"]), 1)
                self.assertEqual(r.stats["queue_full"], 1)
                self.assertEqual(len(r.gate), 1)                                       # the rejected request never queued
                for t in busy:
                    t.cancel()
                await asyncio.gather(*busy, return_exceptions=True)
                st, hd, evs = await waiter
                self.assertEqual(st, 200)                                              # the queued one is served once a lane frees
            finally:
                await close(r, s, m)
        h.run(go())

    def test_waiting_longer_than_max_wait_is_503_with_retry_after(self):
        async def go():
            r, s, m = await pair({"admission": {"max_wait_s": 2.0, "poll_s": 0.2}})
            try:
                busy = [asyncio.ensure_future(h.stream(r.port, long_body("t%d" % i, tokens=4000))) for i in range(2)]
                await asyncio.sleep(0.05)
                st, hd, evs = await h.stream(r.port, h.chat_body(2000, uniq="late", stream=True, max_tokens=5), [("X-Router-Explain", "1")])
                self.assertEqual(st, 503)
                self.assertIn("retry-after", hd)
                self.assertEqual(r.stats["queue_timeout"], 1)
                self.assertEqual(len(r.gate), 0)
                for t in busy:
                    t.cancel()
                await asyncio.gather(*busy, return_exceptions=True)
            finally:
                await close(r, s, m)
        h.run(go())

    def test_client_leaving_the_queue_frees_its_place(self):
        async def go():
            r, s, m = await pair({"admission": {"max_wait_s": 600.0, "poll_s": 0.5}})
            try:
                busy = [asyncio.ensure_future(h.stream(r.port, long_body("l%d" % i))) for i in range(2)]
                await asyncio.sleep(0.1)
                third = asyncio.ensure_future(h.stream(r.port, h.chat_body(2000, uniq="l9", stream=True, max_tokens=8), stop_after=0))
                await asyncio.sleep(0.1)
                self.assertEqual(len(r.gate), 1)
                third.cancel()
                await asyncio.gather(third, return_exceptions=True)
                await asyncio.sleep(0.4)
                self.assertEqual(len(r.gate), 0)
                for t in busy:
                    t.cancel()
                await asyncio.gather(*busy, return_exceptions=True)
            finally:
                await close(r, s, m)
        h.run(go())

    def test_a_forced_request_also_waits_for_its_lane(self):
        async def go():
            r, s, m = await pair({"admission": {"max_wait_s": 600.0}})
            try:
                busy = asyncio.ensure_future(h.stream(r.port, long_body("f1", tokens=2500), [("X-Route-To", "S")]))
                await asyncio.sleep(0.05)
                forced = asyncio.ensure_future(h.stream(r.port, h.chat_body(2000, uniq="f2", stream=True, max_tokens=5), [("X-Route-To", "S")]))
                await asyncio.sleep(0.05)
                self.assertEqual(len(r.gate), 1)                                      # M is idle, but the request is pinned to S
                self.assertEqual(r.backends["M"].in_use(), 0)
                st, hd, _ = await forced
                self.assertEqual((st, hd["x-router-backend"]), (200, "S"))
                await busy
                self.assertEqual(s.m.stats["max_queue"], 0)
            finally:
                await close(r, s, m)
        h.run(go())

    def test_a_failover_target_without_a_lane_is_waited_for(self):
        async def go():
            r, s, m = await pair({"admission": {"max_wait_s": 600.0}, "breaker": {"failure_threshold": 5}})
            try:
                await h.http(s.port, "POST", "/fake/fault", {"mode": "down"})
                busy = asyncio.ensure_future(h.stream(r.port, long_body("fo", tokens=400), [("X-Route-To", "M")]))
                await asyncio.sleep(0.05)
                st, hd, evs = await h.stream(r.port, h.chat_body(2000, uniq="fo2", stream=True, max_tokens=5))
                self.assertEqual((st, hd["x-router-backend"]), (200, "M"))            # S refused, M's only lane was busy: it waited, it did not queue at M
                self.assertEqual(m.m.stats["max_queue"], 0)
                await busy
            finally:
                await close(r, s, m)
        h.run(go())

    def test_state_shows_the_queue(self):
        async def go():
            r, s, m = await pair({"admission": {"max_wait_s": 600.0}})
            try:
                busy = [asyncio.ensure_future(h.stream(r.port, long_body("s%d" % i))) for i in range(2)]
                await asyncio.sleep(0.05)
                third = asyncio.ensure_future(h.stream(r.port, h.chat_body(2000, uniq="s9", stream=True, max_tokens=5)))
                await asyncio.sleep(0.05)
                st, _, body = await h.http(r.port, "GET", "/router/state")
                a = json.loads(body)["admission"]
                self.assertEqual(a["waiting"], 1)
                self.assertEqual(a["per_backend"]["S"], {"in_use": 1, "max_inflight": 1, "waiting": 1, "queue_max": 32})
                for t in busy:
                    t.cancel()
                await asyncio.gather(*busy, return_exceptions=True)
                await third
            finally:
                await close(r, s, m)
        h.run(go())


# ---------------------------------------------------------------------------------------------------------------------------
# cost model: output length, interference, primary bias
# ---------------------------------------------------------------------------------------------------------------------------
class OutputEstimateTests(unittest.TestCase):
    def test_warm_up_replaces_the_start_value_with_the_first_samples(self):
        cfg = cfgmod.load(None, {"backends": {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2"}}})
        t = TokenEstimator(cfg)
        self.assertEqual(t.out_est("interactive", 512), 300)                       # no sample yet: the start value
        t.observe_out("interactive", 512)
        self.assertEqual(t.out_est("interactive", 512), 512)                       # E2: 321 after the first request, 483 after nineteen
        t.observe_out("interactive", 400)
        self.assertEqual(t.out_est("interactive", 1000), 456)                      # running mean ...
        for _ in range(60):
            t.observe_out("interactive", 100)
        self.assertLess(t.out_est("interactive", 1000), 200)                       # ... then the EWMA: follows a drift

    def test_the_old_ewma_is_still_selectable(self):
        cfg = cfgmod.load(None, {"backends": {"S": {"url": "http://s:1"}, "M": {"url": "http://m:2"}}, "tokens": {"output_warmup": False}})
        t = TokenEstimator(cfg)
        t.observe_out("interactive", 512)
        self.assertEqual(t.out_est("interactive", 512), 321)


class InterferenceTests(unittest.TestCase):
    def test_a_runner_that_ends_soon_is_slowed_only_until_then(self):
        cfg, bs, by = setup()
        pf = by["S"].profile
        full = predict(pf, View(k=1, rem=None), 2000, 512, 1.0)                    # sim.py: the runner outlives the new request
        long_ = predict(pf, View(k=1, rem=(10000.0,)), 2000, 512, 1.0)
        short = predict(pf, View(k=1, rem=(40.0,)), 2000, 512, 1.0)
        self.assertEqual(full.interf, long_.interf)
        self.assertLess(short.interf, full.interf - 3.0)                          # 472 tokens x 7 ms less slowdown charged
        self.assertAlmostEqual(short.interf, pf.tok_s(2) * 0 + 1 * short.stall + 40 * (pf.tok_s(2) - pf.tok_s(1)), places=6)

    def test_backend_view_estimates_the_runners_remaining_tokens(self):
        cfg, bs, by = setup()
        s = by["S"]
        r = Inflight("a", "interactive", True, 2000, 2000, 0, 300)
        s.admit(r, 0.0, 1.2, 1.2)
        s.first_token(r, 1.2)
        v = s.view(1.2)
        self.assertEqual(v.rem, (300.0,))                                          # just started: all of the estimate left
        v = s.view(1.2 + 100 * s.profile.tok_s(1))
        self.assertAlmostEqual(v.rem[0], 200.0, delta=1.0)
        v = s.view(1.2 + 400 * s.profile.tok_s(1))                                  # past its estimate: a quarter of it is assumed left
        self.assertAlmostEqual(v.rem[0], 75.0)
        cfg2, _, by2 = setup(policy={"interf_overlap": False})
        self.assertIsNone(by2["S"].view(0.0).rem)


def e2_r2(**ov):
    """the E2 R-chat-0.09 decision of the second request (explain.jsonl): S had one stream that started decoding 4.7 s ago, M was idle; 1937-token prompt"""
    cfg, bs, by = setup(**ov)
    r = Inflight("r1", "interactive", True, 1937, 1937, 0, 300)
    by["S"].admit(r, 0.0, 1.25, 1.25)
    by["S"].first_token(r, 1.25)
    return cfg, bs, decide(cfg, bs, ctx(1937, out=300), 6.3)


class E2RegressionTests(unittest.TestCase):
    def test_the_e2_policy_sent_the_second_chat_request_to_m(self):
        cfg, bs, d = e2_r2(policy={"interf_overlap": False, "primary_bias_frac": 0.0})
        self.assertEqual(d.choice, "M")                                            # explain.jsonl: S cost 12.2 vs M 10.1 (interference 3.4 s charged for one runner)
        self.assertAlmostEqual(d.cands["S"]["pred"]["cost"], 12.23, delta=0.35)
        self.assertAlmostEqual(d.cands["M"]["pred"]["cost"], 10.07, delta=0.05)

    def test_the_fixed_policy_keeps_it_on_s(self):
        cfg, bs, d = e2_r2()
        self.assertEqual(d.choice, "S")                                            # M is 9 % cheaper at best: not worth its slower decode and the extra machine
        self.assertLess(d.cands["S"]["pred"]["interf"], 2.5)

    def test_a_clear_win_still_goes_to_m(self):
        cfg, bs, by = setup()
        load(by["S"], 0.0, decoding=4, waiting_pf=(12.0, 8.0))
        self.assertEqual(decide(cfg, bs, ctx(2000, out=512), 0.0).choice, "M")      # S saturated: M's advantage is 20 s, no hysteresis matters

    def test_the_primary_bias_is_relative(self):
        cfg, bs, by = setup()
        load(by["S"], 0.0, decoding=3)                                              # S k=3: ~25 % dearer than idle M for a 512-token answer
        d = decide(cfg, bs, ctx(2000, out=512), 0.0)
        self.assertEqual(d.choice, "M")
        cfg2, bs2, by2 = setup(policy={"primary_bias_frac": 0.6})
        load(by2["S"], 0.0, decoding=3)
        self.assertEqual(decide(cfg2, bs2, ctx(2000, out=512), 0.0).choice, "S")


class LowLoadDesTests(unittest.TestCase):
    """the E2 chat sweep in the discrete-event simulator, E2 policy vs the fix (6 seeds x 900 s, ~1 s per run)"""

    @classmethod
    def setUpClass(cls):
        import simcompare as sc
        cls.sc, cls.sim = sc, sc.sim
        cls.sim.SEEDS, cls.sim.HORIZON = tuple(range(1, 7)), 900.0

    def run_chat(self, setup_name, rate, slo):
        sims = [self.sim.run_one("chat", rate, self.sim.SETUPS[setup_name], self.sim.PARAM, sd) for sd in self.sim.SEEDS]
        return self.sim.summarize(sims, "chat", slo=slo), self.sc.m_share(sims)

    def test_the_0_09_split_and_goodput(self):
        old, m_old = self.run_chat("Router E2 policy (before the fix)", 0.09, (2.0, 0.10))
        new, m_new = self.run_chat("Router cost (tfrouter default)", 0.09, (2.0, 0.10))
        self.assertGreater(m_old, 0.27)                                            # E2: ~25 % of chat went to M at 0.09 req/s
        self.assertLess(m_new, 0.22)
        self.assertGreater(new["good"], old["good"] * 1.08)                        # the requests kept off M are the ones that missed the 2 s TTFT
        self.assertLessEqual(new["ttft95"], old["ttft95"] + 0.05)

    def test_the_fix_never_loses_to_the_e2_policy_across_the_sweep(self):
        for rate in (0.06, 0.09, 0.12, 0.15):
            for slo in ((4.0, 0.15), (2.0, 0.10)):
                old, _ = self.run_chat("Router E2 policy (before the fix)", rate, slo)
                new, _ = self.run_chat("Router cost (tfrouter default)", rate, slo)
                # at 0.15 the fleet is saturated and the cap leaves a lane idle now and then (the price of never queueing at production): bounded loss only
                tol_good, tol_ttft = (0.94, 1.20) if rate >= 0.15 else (0.98, 1.05)
                self.assertGreaterEqual(new["good"], old["good"] * tol_good, (rate, slo))
                self.assertLessEqual(new["ttft95"], old["ttft95"] * tol_ttft + 0.1, (rate, slo))

    def test_agent_workload_keeps_up_with_the_e2_policy(self):
        """cache-affine multi-turn traffic: a waiting request must keep waiting for the backend that holds its prefix, not take whichever lane is free"""
        for rate in (0.02, 0.05):
            res = []
            for name in ("Router E2 policy (before the fix)", "Router cost (tfrouter default)"):
                sims = [self.sim.run_one("agent", rate, self.sim.SETUPS[name], self.sim.PARAM, sd) for sd in self.sim.SEEDS]
                res.append(self.sim.summarize(sims, "agent"))
            old, new = res
            self.assertGreaterEqual(new["good"], old["good"] * 0.95, rate)
            self.assertLessEqual(new["ttft95"], old["ttft95"] * 1.1 + 0.2, rate)

    def test_admission_control_keeps_both_machines_within_their_slots(self):
        """chat at 0.2 req/s saturates the fleet: without the router queue the machines' own slot queues fill, with it they stay empty and the router queue holds the excess"""
        peak = {"q": 0}
        orig = self.sim.Machine.acquire

        def acquire(m, cont):
            orig(m, cont)
            peak["q"] = max(peak["q"], len(m.admit_q))
        self.sim.Machine.acquire = acquire
        try:
            self.sim.run_one("chat", 0.20, self.sim.SETUPS["Router E2 policy (before the fix)"], self.sim.PARAM, 1, 600.0)
            self.assertGreater(peak["q"], 0)
            peak["q"] = 0
            sm = self.sim.run_one("chat", 0.20, self.sim.SETUPS["Router cost (tfrouter default)"], self.sim.PARAM, 1, 600.0)
            self.assertEqual(peak["q"], 0)
            self.assertGreater(sm.rt.gate.seq, 20)                                 # ... the excess waited in the router instead
        finally:
            self.sim.Machine.acquire = orig


if __name__ == "__main__":
    unittest.main()
