"""end-to-end tests: the real router (in process) in front of fake backends (in process, fast-forwarded 200x)"""
import asyncio
import json
import os
import tempfile
import unittest

from tests.helpers import SPEED, chat_body, http, run, start_fake, start_router, stream


class Env:
    """async context: two fakes + a router"""

    def __init__(self, router_ov=None, s_kw=None, m_kw=None):
        self.router_ov, self.s_kw, self.m_kw = router_ov, s_kw or {}, m_kw or {}

    async def __aenter__(self):
        self.S = await start_fake("S", **self.s_kw)
        self.M = await start_fake("M", **self.m_kw)
        self.R = await start_router({"S": self.S, "M": self.M}, self.router_ov)
        return self

    async def __aexit__(self, *exc):
        await self.R.stop()
        await self.S.close()
        await self.M.close()

    async def chat(self, n, **kw):
        extra = kw.pop("headers", ())
        b = chat_body(n, **kw)
        return await http(self.R.port, "POST", "/v1/chat/completions", b, extra)

    async def state(self):
        st, h, body = await http(self.R.port, "GET", "/router/state")
        return json.loads(body)

    async def last_decision(self):
        st, h, body = await http(self.R.port, "GET", "/router/explain")
        return json.loads(body)[-1]


class BasicTests(unittest.TestCase):
    def test_models_health_and_nonstream_chat(self):
        async def go():
            async with Env() as e:
                st, h, body = await http(e.R.port, "GET", "/v1/models")
                self.assertEqual(st, 200)
                ids = [m["id"] for m in json.loads(body)["data"]]
                self.assertIn("DeepSeek-V4.1-Flash", ids)
                self.assertIn("DeepSeek-V4.1-Flash:background", ids)
                st, h, body = await http(e.R.port, "GET", "/health")
                self.assertEqual(st, 200)
                st, h, body = await e.chat(800, uniq="p")
                self.assertEqual(st, 200)
                o = json.loads(body)
                self.assertEqual(o["usage"]["completion_tokens"], 20)
                self.assertEqual(o["model"], "DeepSeek-V4.1-Flash-1017")     # the backend's own name: the router rewrote the request's model
                self.assertEqual(h["x-router-backend"], "S")
                st, h, body = await http(e.R.port, "POST", "/v1/completions", {"model": "x", "prompt": "hello " * 200, "max_tokens": 5})
                self.assertEqual(st, 200)
                self.assertEqual(json.loads(body)["usage"]["completion_tokens"], 5)
        run(go())

    def test_stream_passthrough_is_complete_and_ordered(self):
        async def go():
            async with Env() as e:
                st, h, evs = await stream(e.R.port, chat_body(500, uniq="s", stream=True, max_tokens=40))
                self.assertEqual(st, 200)
                self.assertEqual(h["x-router-backend"], "S")
                self.assertEqual(evs[-1][1], b"[DONE]")
                chunks = [json.loads(p) for _, p in evs[:-1]]
                text = "".join((c["choices"][0].get("delta") or {}).get("content") or "" for c in chunks if c["choices"])
                self.assertEqual(len(text), sum(max(1, int(round((i + 1) * 3.3)) - int(round(i * 3.3))) for i in range(40)))
                self.assertIn("usage", chunks[-1])                       # the final event carries usage (docs/api.md)
                self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 40)
                self.assertGreater(len(evs), 10)                          # several events: not buffered into one
        run(go())

    def test_include_usage_event_style(self):
        async def go():
            async with Env() as e:
                b = chat_body(500, uniq="u", stream=True, max_tokens=10, extra={"stream_options": {"include_usage": True}})
                st, h, evs = await stream(e.R.port, b)
                self.assertEqual(st, 200)
                last = json.loads(evs[-2][1])
                self.assertEqual(last["choices"], [])
                self.assertEqual(last["usage"]["completion_tokens"], 10)
                s = await e.state()
                self.assertEqual(s["backends"][0]["counts"]["ok"], 1)
        run(go())

    def test_bad_requests(self):
        async def go():
            async with Env() as e:
                st, _, body = await http(e.R.port, "POST", "/v1/chat/completions", b"{not json")
                self.assertEqual(st, 400)
                st, _, _ = await http(e.R.port, "POST", "/v1/chat/completions", {"model": "x"})
                self.assertEqual(st, 400)
                st, _, _ = await http(e.R.port, "GET", "/v1/nonexistent")
                self.assertEqual(st, 404)
                st, _, _ = await e.chat(500, headers=[("X-Request-Class", "bogus")])
                self.assertEqual(st, 400)
        run(go())

    def test_client_api_key(self):
        async def go():
            with tempfile.NamedTemporaryFile("w", delete=False) as f:
                f.write("# keys\nweb: sk-client-1\n")
            try:
                async with Env({"client_api_key_file": f.name}) as e:
                    st, _, _ = await e.chat(500, uniq="k")
                    self.assertEqual(st, 401)
                    st, _, _ = await e.chat(500, uniq="k", headers=[("Authorization", "Bearer sk-client-1")])
                    self.assertEqual(st, 200)
                    st, _, _ = await http(e.R.port, "GET", "/health")
                    self.assertEqual(st, 200)                              # health stays open
            finally:
                os.unlink(f.name)
        run(go())

    def test_backend_api_keys_from_files(self):
        async def go():
            with tempfile.NamedTemporaryFile("w", delete=False) as fs, tempfile.NamedTemporaryFile("w", delete=False) as fm:
                fs.write("head: sk-sparks\n")
                fm.write("sk-m5\n")
            try:
                async with Env({"backends": {"S": {"api_key_file": fs.name}, "M": {"api_key_file": fm.name}}}, s_kw={"api_key_file": fs.name}, m_kw={"api_key_file": fm.name}) as e:
                    st, h, body = await e.chat(500, uniq="a")
                    self.assertEqual((st, h["x-router-backend"]), (200, "S"))
                    st, h, body = await e.chat(500, uniq="b", headers=[("X-Route-To", "M")])
                    self.assertEqual((st, h["x-router-backend"]), (200, "M"))
                # a wrong key on S: the backend's 401 is the client's answer (and no failover: the other backend would say the same)
                with tempfile.NamedTemporaryFile("w", delete=False) as bad:
                    bad.write("wrong\n")
                try:
                    async with Env({"backends": {"S": {"api_key_file": bad.name}, "M": {"api_key_file": fm.name}}}, s_kw={"api_key_file": fs.name}, m_kw={"api_key_file": fm.name}) as e:
                        st, h, body = await e.chat(500, uniq="c")
                        self.assertEqual(st, 401)
                        s = await e.state()
                        self.assertEqual(s["backends"][0]["breaker"]["state"], "closed")   # a client error is not a backend failure
                finally:
                    os.unlink(bad.name)
            finally:
                os.unlink(fs.name)
                os.unlink(fm.name)
        run(go())


class RoutingTests(unittest.TestCase):
    def test_context_limit_sends_long_prompts_to_primary(self):
        async def go():
            async with Env() as e:
                st, h, body = await e.chat(52000, uniq="L", max_tokens=4)
                self.assertEqual((st, h["x-router-backend"]), (200, "S"))
                d = await e.last_decision()
                self.assertEqual(d["candidates"]["M"]["excluded"], "context")
                st, h, body = await e.chat(1_100_000, uniq="X", max_tokens=4)
                self.assertEqual(st, 413)
        run(go())

    def test_prefix_affinity_keeps_a_conversation_on_its_backend(self):
        async def go():
            async with Env() as e:
                msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1 " + "w" * 12000}]
                b = {"model": "m", "messages": msgs, "max_tokens": 20}
                st, h, body = await http(e.R.port, "POST", "/v1/chat/completions", b, [("X-Route-To", "M")])     # turn 1 lands on M (e.g. overflow)
                self.assertEqual(h["x-router-backend"], "M")
                reply = json.loads(body)["choices"][0]["message"]["content"]
                msgs2 = msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": "q2 short"}]
                st, h, body = await http(e.R.port, "POST", "/v1/chat/completions", dict(b, messages=msgs2, max_tokens=20))
                d = await e.last_decision()
                self.assertGreater(d["cached"]["M"], 3000)
                self.assertEqual(d["cached"]["S"], 0)
                self.assertEqual(h["x-router-backend"], "M", d)                 # the cache hit (rows ~ 10 vs 3.6k) outweighs S's default
                usage = json.loads(body)["usage"]
                self.assertGreater(usage["prompt_tokens_details"]["cached_tokens"], 3000)   # and the backend really had it
        run(go())

    def test_session_header_affinity(self):
        async def go():
            async with Env() as e:
                st, h, _ = await e.chat(3000, uniq="z", headers=[("X-Route-To", "M"), ("X-Session-Id", "c1")])
                self.assertEqual(h["x-router-backend"], "M")
                st, h, _ = await e.chat(3300, uniq="y", max_tokens=10, headers=[("X-Session-Id", "c1")])   # different text, same session
                d = await e.last_decision()
                self.assertGreater(d["cached"]["M"], 2000)
        run(go())

    def test_class_by_model_suffix_and_header(self):
        async def go():
            async with Env() as e:
                await e.chat(500, uniq="c1", model="DeepSeek-V4.1-Flash:background")
                self.assertEqual((await e.last_decision())["class"], "background")
                await e.chat(500, uniq="c2", model="DeepSeek-V4.1-Flash:background", headers=[("X-Request-Class", "interactive")])
                self.assertEqual((await e.last_decision())["class"], "interactive")      # the header wins
                await e.chat(500, uniq="c3")
                self.assertEqual((await e.last_decision())["class"], "interactive")
        run(go())

    def test_background_goes_to_secondary(self):
        async def go():
            async with Env() as e:
                st, h, _ = await e.chat(9000, uniq="bg", max_tokens=8, model="DeepSeek-V4.1-Flash:background")
                self.assertEqual(h["x-router-backend"], "M")
        run(go())

    def test_overflow_when_primary_is_busy(self):
        """5 long-ish prompts at once: S's FIFO prefill queue builds, the cost model moves the tail to M"""
        async def go():
            async with Env() as e:
                reqs = [e.chat(8000, uniq="o%d" % i, max_tokens=30) for i in range(5)]
                res = await asyncio.gather(*reqs)
                backends = [h["x-router-backend"] for _, h, _ in res]
                self.assertTrue(all(st == 200 for st, _, _ in res))
                self.assertIn("M", backends)
                self.assertIn("S", backends)
                self.assertEqual(backends[0], "S")
        run(go())

    def test_dry_run_and_explain_header(self):
        async def go():
            async with Env({"mode": {"dry_run": True}}) as e:
                st, h, body = await e.chat(2500, uniq="d")
                o = json.loads(body)["router_decision"]
                self.assertEqual(st, 200)
                self.assertEqual(o["choice"], "S")
                self.assertIn("pred", o["candidates"]["S"])
                self.assertEqual(e.S.m.stats["requests"] + e.M.m.stats["requests"], 0)       # no backend was contacted
            async with Env() as e:
                st, h, body = await e.chat(500, uniq="x", headers=[("X-Router-Explain", "1")])
                d = json.loads(h["x-router-decision"])
                self.assertEqual(d["choice"], "S")
                self.assertEqual(d["rule"], "min-cost")
        run(go())

    def test_explain_log_file(self):
        async def go():
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
                pass
            try:
                async with Env({"log": {"explain": True, "explain_file": f.name}}) as e:
                    await e.chat(500, uniq="l1")
                    await e.chat(4500, uniq="l2")
                lines = [json.loads(l) for l in open(f.name) if l.strip()]
                self.assertEqual(len(lines), 2)
                self.assertEqual(lines[0]["id"], "r1")
                self.assertIn("M", lines[1]["candidates"])
            finally:
                os.unlink(f.name)
        run(go())

    def test_shadow_mode_logs_but_always_uses_the_named_backend(self):
        async def go():
            async with Env({"mode": {"shadow": "S"}}) as e:
                st, h, _ = await e.chat(9000, uniq="sh", max_tokens=8, model="DeepSeek-V4.1-Flash:background")     # policy would say M
                d = await e.last_decision()
                self.assertEqual(d["choice"], "M")
                self.assertEqual(d["shadow"], "S")
                self.assertEqual(h["x-router-backend"], "S")
        run(go())

    def test_exact_tokenizer_used_near_a_boundary(self):
        async def go():
            async with Env() as e:
                await e.chat(40400, uniq="t", max_tokens=4)            # within 12 % of M's 40,960 context: ask the tokenizer
                d = await e.last_decision()
                self.assertEqual(d["n_src"], "tokenize")
                self.assertAlmostEqual(d["n_in"], 40400, delta=30)     # the fake tokenizer's count (ours would be 3.5 chars/token: ~6 % off)
                s = await e.state()
                self.assertGreaterEqual(s["tokens"]["exact_calls"], 1)
                await e.chat(900, uniq="t2")
                self.assertEqual((await e.last_decision())["n_src"], "estimate")
        run(go())

    def test_calibrates_chars_per_token_and_prefill_scale(self):
        async def go():
            async with Env(m_kw={"prefill_scale": 1.5}) as e:
                for i in range(12):
                    await e.chat(3000 + 100 * i, uniq="cal%d" % i, max_tokens=8)
                    await http(e.R.port, "POST", "/v1/chat/completions", chat_body(3000 + 100 * i, uniq="calm%d" % i, max_tokens=8), [("X-Route-To", "M")])
                s = await e.state()
                self.assertAlmostEqual(s["tokens"]["chars_per_token"], 3.3, delta=0.25)     # fake tokenizer: 3.3 chars/token (router started at 3.5)
                by = {b["name"]: b for b in s["backends"]}
                self.assertAlmostEqual(by["M"]["ewma"]["pf_scale"], 1.5, delta=0.2)          # M was 1.5x slower than its table
                self.assertAlmostEqual(by["S"]["ewma"]["pf_scale"], 1.0, delta=0.15)
                self.assertIsNotNone(by["S"]["ewma"]["prefill_tps"])
        run(go())

    def test_nonstream_without_timings_block_still_works(self):
        async def go():
            async with Env(s_kw={"no_timings": True}) as e:
                for i in range(3):
                    st, h, body = await e.chat(2500, uniq="nt%d" % i, max_tokens=64)
                    self.assertEqual(st, 200)
                s = await e.state()
                self.assertEqual(s["backends"][0]["counts"]["ok"], 3)
        run(go())

    def test_external_load_is_sensed(self):
        async def go():
            async with Env({"health": {"interval_s": 1.0, "load_poll_s": 1.0}}) as e:
                # somebody else (not through the router) streams on S
                other = [asyncio.ensure_future(stream(e.S.port, chat_body(400, uniq="ext%d" % i, stream=True, max_tokens=6000, model="DeepSeek-V4.1-Flash-1017"))) for i in range(3)]
                await asyncio.sleep(0.15)                                 # ~30 virtual seconds: several load polls
                s = await e.state()
                self.assertGreaterEqual(s["backends"][0]["external_active"], 2)
                await e.chat(500, uniq="mine")
                d = await e.last_decision()
                self.assertGreaterEqual(d["candidates"]["S"]["view"]["k"], 2)
                for t in other:
                    t.cancel()
        run(go())


class FailureTests(unittest.TestCase):
    def test_cancellation_propagates_to_the_backend(self):
        async def go():
            async with Env() as e:
                b = chat_body(800, uniq="cx", stream=True, max_tokens=3000)
                st, h, evs = await stream(e.R.port, b, stop_after=3)       # the client hangs up after 3 events
                self.assertEqual(st, 200)
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    if e.S.m.stats["cancelled"]:
                        break
                self.assertEqual(e.S.m.stats["cancelled"], 1)
                self.assertEqual(len(e.S.m.active), 0)
                self.assertEqual(e.S.m.slots_used, 0)                      # slot released
                s = await e.state()
                self.assertEqual(s["backends"][0]["counts"]["cancelled"], 1)
                self.assertEqual(s["backends"][0]["inflight"], 0)
                self.assertEqual(s["stats"]["client_cancels"], 1)
                self.assertEqual(s["backends"][0]["breaker"]["state"], "closed")
        run(go())

    def test_cancellation_during_prefill(self):
        async def go():
            async with Env() as e:
                reader_task = asyncio.ensure_future(http(e.R.port, "POST", "/v1/chat/completions", chat_body(100000, uniq="pf", stream=True, max_tokens=50)))
                await asyncio.sleep(0.06)                                   # ~12 virtual s of a ~50 s prefill, then drop
                reader_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await reader_task
                for _ in range(200):
                    await asyncio.sleep(0.02)
                    if e.S.m.stats["cancelled"]:
                        break
                self.assertEqual(e.S.m.stats["cancelled"] + e.M.m.stats["cancelled"], 1)
                self.assertEqual(len(e.S.m.queue) + len(e.M.m.queue), 0)
        run(go())

    def test_failover_when_a_backend_is_down_and_breaker_cycle(self):
        async def go():
            async with Env({"breaker": {"failure_threshold": 2, "cooldown_s": 6.0}, "health": {"interval_s": 3.0}}) as e:
                await http(e.S.port, "POST", "/fake/fault", {"mode": "down"})
                st, h, _ = await e.chat(1500, uniq="f1")
                self.assertEqual((st, h["x-router-backend"]), (200, "M"))         # S refused: retried on M
                st, h, _ = await e.chat(1500, uniq="f2")
                self.assertEqual((st, h["x-router-backend"]), (200, "M"))
                s = await e.state()
                self.assertEqual(s["backends"][0]["breaker"]["state"], "open")
                d = None
                st, h, _ = await e.chat(1500, uniq="f3")
                self.assertEqual(h["x-router-backend"], "M")
                self.assertEqual((await e.last_decision())["candidates"]["S"]["excluded"], "breaker-open")
                self.assertGreaterEqual(s["stats"]["failovers"], 1)
                # S comes back: after the cooldown a trial request closes the breaker
                await http(e.S.port, "POST", "/fake/fault", {"mode": "ok"})
                await asyncio.sleep(0.12)                                          # 24 virtual seconds
                st, h, _ = await e.chat(1500, uniq="f4")
                st, h, _ = await e.chat(1500, uniq="f5")
                s = await e.state()
                self.assertEqual(s["backends"][0]["breaker"]["state"], "closed")
                self.assertEqual(h["x-router-backend"], "S")
        run(go())

    def test_failover_on_503_and_429(self):
        async def go():
            async with Env({"health": {"interval_s": 100000.0}}) as e:
                await http(e.S.port, "POST", "/fake/fault", {"mode": "err503"})
                st, h, _ = await e.chat(1500, uniq="g1")
                self.assertEqual((st, h["x-router-backend"]), (200, "M"))
                before = (await e.state())["backends"][0]["breaker"]["fails"]
                await http(e.S.port, "POST", "/fake/fault", {"mode": "err429"})
                st, h, _ = await e.chat(1500, uniq="g2")
                self.assertEqual((st, h["x-router-backend"]), (200, "M"))
                s = await e.state()
                self.assertGreaterEqual(before, 1)                                                  # the 503 counted ...
                self.assertEqual(s["backends"][0]["breaker"]["fails"], before)                      # ... the 429 (busy) did not
        run(go())

    def test_streaming_failover_before_first_token(self):
        async def go():
            async with Env() as e:
                await http(e.S.port, "POST", "/fake/fault", {"mode": "err503"})
                st, h, evs = await stream(e.R.port, chat_body(1200, uniq="sf", stream=True, max_tokens=12))
                self.assertEqual((st, h["x-router-backend"]), (200, "M"))
                self.assertEqual(evs[-1][1], b"[DONE]")
        run(go())

    def test_both_backends_down_is_503(self):
        async def go():
            async with Env({"breaker": {"failure_threshold": 1}}) as e:
                await http(e.S.port, "POST", "/fake/fault", {"mode": "down"})
                await http(e.M.port, "POST", "/fake/fault", {"mode": "down"})
                st, h, body = await e.chat(1200, uniq="bd")
                self.assertEqual(st, 503)
                self.assertIn("Retry-After", {k.title(): v for k, v in h.items()})
                st, h, body = await e.chat(1200, uniq="bd2")
                self.assertEqual(st, 503)
                st, _, _ = await http(e.R.port, "GET", "/health")
                self.assertEqual(st, 503)
        run(go())

    def test_mid_stream_failure_ends_with_an_error_event(self):
        async def go():
            async with Env() as e:
                await http(e.S.port, "POST", "/fake/fault", {"mode": "drop_mid", "after": 3})
                st, h, evs = await stream(e.R.port, chat_body(1200, uniq="mid", stream=True, max_tokens=200), [("X-Route-To", "S")])
                self.assertEqual(st, 200)
                self.assertIn(b"error", evs[-2][1])
                self.assertEqual(evs[-1][1], b"[DONE]")
                s = await e.state()
                self.assertEqual(s["backends"][0]["breaker"]["fails"], 1)
        run(go())

    def test_hung_backend_times_out_then_fails_over(self):
        async def go():
            async with Env({"failover": {"first_byte_timeout_s": 5.0, "first_byte_per_ptok_s": 0.0}}) as e:
                await http(e.S.port, "POST", "/fake/fault", {"mode": "hang"})
                st, h, _ = await e.chat(1200, uniq="hang")
                self.assertEqual((st, h["x-router-backend"]), (200, "M"))
        run(go())

    def test_health_probe_opens_the_breaker_of_a_dead_backend(self):
        async def go():
            async with Env({"health": {"interval_s": 2.0}, "breaker": {"failure_threshold": 2, "cooldown_s": 100.0}}) as e:
                await http(e.S.port, "POST", "/fake/fault", {"mode": "down"})
                await asyncio.sleep(0.25)
                s = await e.state()
                self.assertEqual(s["backends"][0]["breaker"]["state"], "open")
                self.assertFalse(s["backends"][0]["health"]["ok"])
                st, h, _ = await e.chat(1200, uniq="hp")
                self.assertEqual(h["x-router-backend"], "M")
        run(go())


if __name__ == "__main__":
    unittest.main()
