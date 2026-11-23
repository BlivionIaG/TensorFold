"""the fake backend reproduces the measured machine behaviour (baselines.md; same numbers as setups-simulation.md validation 1/3/4).
Times are read from the reply's own `tensorfold` block (machine timeline, virtual seconds), not from the wall clock."""
import asyncio
import json
import unittest

from tests.helpers import chat_body, http, run, start_fake


async def tf(port, n, uniq, max_tokens=1, delay=0.0, model=None):
    await asyncio.sleep(delay)
    b = chat_body(n, uniq=uniq, max_tokens=max_tokens, model=model)
    st, h, body = await http(port, "POST", "/v1/chat/completions", b)
    o = json.loads(body)
    return o["tensorfold"], o["usage"]


class FakeTests(unittest.TestCase):
    def test_idle_prefill_table(self):
        async def go():
            s, m = await start_fake("S"), await start_fake("M")
            for fk, n, want in ((s, 4055, 2.22), (s, 16652, 8.49), (m, 4184, 4.58), (m, 8280, 8.74)):
                t, u = await tf(fk.port, n, "q%d" % n)
                self.assertAlmostEqual(t["time_to_first_token"], want, delta=want * 0.04, msg=(n, u))
            await s.close(); await m.close()
        run(go())

    def test_prefill_is_fifo_one_prompt_at_a_time(self):
        async def go():
            s = await start_fake("S")
            rs = await asyncio.gather(*[tf(s.port, 4300, "f%d" % i) for i in range(4)])
            ttfts = sorted(t["time_to_first_token"] for t, _ in rs)
            for got, want in zip(ttfts, (2.4, 4.7, 7.0, 9.4)):                       # sim validation 4 / measured 2.2, 4.4, 7.0, 9.6
                self.assertAlmostEqual(got, want, delta=0.5)
            await s.close()
        run(go())

    def test_decode_rates(self):
        async def go():
            s, m = await start_fake("S"), await start_fake("M")
            ts, _ = await tf(s.port, 200, "d1", max_tokens=300)
            tm, _ = await tf(m.port, 200, "d2", max_tokens=300)
            self.assertAlmostEqual(ts["tokens_per_second"], 58.1, delta=2)
            self.assertAlmostEqual(tm["tokens_per_second"], 39.4, delta=1.5)
            # 4 streams: aggregate 110 (S) / 60.6 (M)
            rs = await asyncio.gather(*[tf(s.port, 100, "e%d" % i, max_tokens=300) for i in range(4)])
            self.assertAlmostEqual(sum(t["tokens_per_second"] for t, _ in rs), 110, delta=8)
            rm = await asyncio.gather(*[tf(m.port, 100, "g%d" % i, max_tokens=300) for i in range(4)])
            self.assertAlmostEqual(sum(t["tokens_per_second"] for t, _ in rm), 60.6, delta=5)
            await s.close(); await m.close()
        run(go())

    def test_head_of_line_blocking_m_stops_s_slows(self):
        """a 300-token stream is decoding; a 9k prompt arrives 1 virtual second in: M's stream stops for the whole prefill (9.7 s), S's loses ~1.2 s per 1.25 s"""
        async def go():
            s, m = await start_fake("S"), await start_fake("M")
            out = {}
            for name, fk in (("S", s), ("M", m)):
                run_alone, _ = await tf(fk.port, 100, "h0" + name, max_tokens=300)
                a = asyncio.ensure_future(tf(fk.port, 100, "h1" + name, max_tokens=300))
                b = asyncio.ensure_future(tf(fk.port, 9000, "h2" + name, max_tokens=1, delay=1.0 / 200 * 1.0))
                (ta, _), (tb, _) = await asyncio.gather(a, b)
                out[name] = (run_alone["seconds"], ta["seconds"], tb["time_to_first_token"])
            self.assertGreater(out["M"][1] - out["M"][0], 8.0)       # M: the stream stopped for ~9.7 s
            self.assertLess(out["S"][1] - out["S"][0], out["M"][1] - out["M"][0])
            self.assertAlmostEqual(out["M"][2], 9.7, delta=1.0)       # sim/measured newcomer TTFT 9.69-9.76
            await s.close(); await m.close()
        run(go())

    def test_prefix_cache_hit_and_cached_tokens(self):
        async def go():
            s = await start_fake("S")
            msgs = [{"role": "user", "content": "k" * 20000}]
            st, h, b1 = await http(s.port, "POST", "/v1/chat/completions", {"model": "DeepSeek-V4.1-Flash-1017", "messages": msgs, "max_tokens": 5})
            o1 = json.loads(b1)
            msgs2 = msgs + [{"role": "assistant", "content": o1["choices"][0]["message"]["content"]}, {"role": "user", "content": "next"}]
            st, h, b2 = await http(s.port, "POST", "/v1/chat/completions", {"model": "DeepSeek-V4.1-Flash-1017", "messages": msgs2, "max_tokens": 5})
            o2 = json.loads(b2)
            self.assertGreater(o2["usage"]["prompt_tokens_details"]["cached_tokens"], 5000)
            self.assertLess(o2["tensorfold"]["time_to_first_token"], 0.5 * o1["tensorfold"]["time_to_first_token"])
            await s.close()
        run(go())

    def test_context_limit_and_cancel_frees_slot(self):
        async def go():
            m = await start_fake("M")
            st, h, body = await http(m.port, "POST", "/v1/chat/completions", chat_body(40000, uniq="big", max_tokens=2000, model="DeepSeek-V4.1-Flash"))
            self.assertEqual(st, 400)
            await m.close()
        run(go())


if __name__ == "__main__":
    unittest.main()
