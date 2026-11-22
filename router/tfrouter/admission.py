"""Admission control and late binding: the router never has more than `max_inflight` (default: the slot count) requests in flight on a backend.

E2 (2026-11-11): the router sent every request it had a prediction for straight to a backend and let the backend queue the excess; at chat 0.15 req/s
it drove production (S) well past its 4 slots and the Sparks stopped answering.  Now:

  * a request whose BEST backend (the policy's first choice, which already prices that backend's slot wait) has a free lane goes at once, unless an older /
    higher-priority waiter wants the same backend;
  * otherwise it waits HERE (interactive class first, then age), bound to no backend: every time a lane frees (or after `poll_s`) it is decided again on fresh
    state, so it lands on whichever backend is best at that moment, not on the one that was best when it arrived.  It never takes a worse backend just because
    that one has a lane (a long prompt kept off M by the guard, a conversation whose cache is on S);
  * a request whose eligible backends all have `queue_max` waiters gets 503 + Retry-After at once, a request that waited `max_wait_s` gets 503 + Retry-After;
  * lanes count the router's own requests plus other clients' requests the server reports (`Backend.in_use`).

`WaitQueue` and the lane predicates are plain python (the discrete-event comparison in tools/simcompare.py uses them); `bind` / `wait_lane` are the live parts.
"""
import asyncio
import contextlib
import math

INF = float("inf")


class WaitQueue:
    """waiters sorted by (priority, arrival); priority 0 = interactive classes, 1 = the rest.
    A waiter is [prio, seq, event, eligible backend names (queue accounting), the backend it is waiting for now (its best choice)]."""

    def __init__(self):
        self.waiters = []
        self.seq = 0

    def __len__(self):
        return len(self.waiters)

    def add(self, prio, elig=(), want=None):
        self.seq += 1
        w = [prio, self.seq, asyncio.Event(), frozenset(elig), frozenset(want if want is not None else elig)]
        self.waiters.append(w)
        self.waiters.sort(key=lambda x: (x[0], x[1]))
        return w

    def remove(self, w):
        with contextlib.suppress(ValueError):
            self.waiters.remove(w)

    def kick(self):
        """a lane freed: wake every waiter, in order; each decides again on fresh state"""
        for w in list(self.waiters):
            w[2].set()

    async def wait(self, w, timeout):
        """True: kicked; False: timed out.  The waiter's event is re-armed for the next round."""
        try:
            await asyncio.wait_for(w[2].wait(), timeout)
            w[2].clear()
            return True
        except asyncio.TimeoutError:
            return False

    def depth(self, name):
        """waiters that could be served by backend `name`"""
        return sum(1 for w in self.waiters if name in w[3])

    def ahead(self, prio, seq):
        """{backend name: waiters ahead of (prio, seq) that are waiting for it}"""
        out = {}
        for w in self.waiters:
            if (w[0], w[1]) < (prio, seq):
                for n in w[4]:
                    out[n] = out.get(n, 0) + 1
        return out

    def blocked(self, prio, seq, names):
        """True if a waiter ahead of (prio, seq) is waiting for a lane of any of `names`: it has the right to it first"""
        return any((w[0], w[1]) < (prio, seq) and not names.isdisjoint(w[4]) for w in self.waiters)


def prio_of(cfg, cls):
    return 0 if cfg["policy"]["classes"].get(cls, {}).get("slo") else 1


def decode_backend(router, dec, name):
    return router.backends[dec.paths[name].backend] if name in dec.paths else router.backends[name]


def eligible(router, dec):
    """names of the backends whose lane any candidate of the decision needs"""
    return frozenset(decode_backend(router, dec, n).name for n in dec.ranking)


def best_backend(router, dec):
    """the backend of the decision's first choice"""
    return decode_backend(router, dec, dec.ranking[0])


def ready(router, dec, gate, prio, seq):
    """the first choice has a free lane and no older / higher-priority waiter wants it"""
    b = best_backend(router, dec)
    return b.lane_free() and not gate.blocked(prio, seq, {b.name})


def retry_after(router, dec, now):
    """seconds a rejected client should wait: the soonest predicted slot wait among the eligible backends, within the configured bounds"""
    a = router.cfg["admission"]
    waits = [router.backends[n].view(now).slot_wait for n in eligible(router, dec)] or [0.0]
    return int(math.ceil(min(a["retry_after_max_s"], max(a["retry_after_min_s"], min(waits) + 1.0))))


def reject(router, dec, rule, msg, now):
    dec.status, dec.rule, dec.error, dec.retry_after = 503, rule, msg, retry_after(router, dec, now)
    return dec


async def bind(router, ctx, dec, get_client, redecide):
    """-> (decision, seconds waited).  Returns at once when a lane is free; else waits in the queue (see module doc) and decides again on every wake-up.
    The returned decision's first choice has a free lane, or its status is 503 (queue full / timeout) or whatever `redecide` said (413, no backend)."""
    a = router.cfg["admission"]
    if not a["enabled"] or dec.status != 200:
        return dec, 0.0
    gate, clock = router.gate, router.clock
    prio = prio_of(router.cfg, ctx.cls)
    if ready(router, dec, gate, prio, INF):
        return dec, 0.0
    elig = eligible(router, dec)
    if all(gate.depth(n) >= router.backends[n].queue_max for n in elig):
        router.stats["queue_full"] += 1
        return reject(router, dec, "queue-full", "every eligible backend has %d requests waiting" % max(router.backends[n].queue_max for n in elig), clock.now()), 0.0
    client = get_client()
    w = gate.add(prio, elig, {best_backend(router, dec).name})
    router.stats["held"] += 1
    t0 = clock.now()
    try:
        while True:
            left = float(a["max_wait_s"]) - (clock.now() - t0)
            if left <= 0:
                router.stats["queue_timeout"] += 1
                return reject(router, dec, "queue-timeout", "waited %.0f s for a free slot" % (clock.now() - t0), clock.now()), clock.now() - t0
            poll = clock.wall(min(float(a["poll_s"]), left))
            await router.race(gate.wait(w, poll), client, poll + 5.0)
            now = clock.now()
            ctx.sm_off = router.pd.off_until > now
            ctx.ahead = gate.ahead(w[0], w[1])
            dec = redecide(now)
            if dec.status != 200:
                return dec, now - t0
            w[3], w[4] = eligible(router, dec), frozenset({best_backend(router, dec).name})
            if ready(router, dec, gate, w[0], w[1]):
                return dec, now - t0
    finally:
        gate.remove(w)


async def wait_lane(router, ctx, b, client):
    """a failover target without a free lane: wait for one (same queue, same priority rules).  False if `max_wait_s` passed first."""
    a = router.cfg["admission"]
    clock, gate = router.clock, router.gate
    prio = prio_of(router.cfg, ctx.cls)
    w = gate.add(prio, {b.name})
    router.stats["held"] += 1
    t0 = clock.now()
    try:
        while True:
            if b.lane_free() and not gate.blocked(w[0], w[1], {b.name}):
                return True
            left = float(a["max_wait_s"]) - (clock.now() - t0)
            if left <= 0 or not b.available(clock.now()):
                router.stats["queue_timeout"] += 1
                return False
            poll = clock.wall(min(float(a["poll_s"]), left))
            await router.race(gate.wait(w, poll), client, poll + 5.0)
    finally:
        gate.remove(w)
