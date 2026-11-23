"""Prefix affinity: which backend already holds (the KV of) the start of this prompt.

A request's text is rendered canonically (role + content per message, or the raw `prompt`) and hashed incrementally.  Anchors are the
digests of the text prefix at every message boundary and at character positions every `stride_chars` (then +12.5 % steps), so two
prompts that share a long head (system prompt, document, earlier turns) share anchor digests.  The index maps anchor -> {backend: tokens held}.
Lookup = the longest anchor present for a backend; the rows that backend can skip are that anchor's token count.  A request's chain is seeded
with what changes the rendered prefix besides the messages (tools, chat_template_kwargs; `seed`).  Capacity model: a backend's prompt cache is a
pool of `budget` tokens, so an entry is believed gone once more than `budget` uncached tokens were prefilled on that backend after it was
last touched (`churn`; S without spill tier: ~800k tokens [INFERENCE from the 1M-position pool, SessionTier], measured only as "present after
0 churn, gone after 1.25M"); `ttl_s` is a safety net.  `invalidate` drops a chain when a reply's cached_tokens disproves it.

Explicit sessions (X-Session-Id) are a second, unverified source: session -> (backend, tokens of the last turn), cached = min(tokens, n_in - 1).
"""
import hashlib
import itertools
from collections import OrderedDict


class Anchors:
    """digests of the prefixes of one request: list of (chars, digest) ascending, plus total chars"""
    __slots__ = ("points", "total_chars", "hasher")

    def __init__(self, points, total_chars, hasher):
        self.points, self.total_chars, self.hasher = points, total_chars, hasher


def render_chat(messages):
    """yield (text_piece, is_boundary) for a chat message list; content may be a string or a list of {type:text,text}"""
    for m in messages:
        if not isinstance(m, dict):
            yield str(m), True
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
        elif c is None:
            c = ""
        tc = m.get("tool_calls")
        extra = ""
        if tc:
            extra = repr(tc)
        yield "\x1e" + str(m.get("role", "")) + "\x1f", False
        yield str(c) + extra, True


class PrefixIndex:
    def __init__(self, cfg):
        a = cfg["affinity"]
        self.enabled = a["enabled"]
        self.ttl = float(a["ttl_s"])
        self.max_entries = int(a["max_entries"])
        self.stride = int(a["stride_chars"])
        self.linear_until = int(a["linear_until_chars"])
        self.growth = float(a["growth"])
        self.min_tokens = int(a["min_prefix_tokens"])
        self.record_reply = bool(a["record_reply"])
        self.table = OrderedDict()      # (chars, digest) -> {backend: (tokens, ts, churn_at_touch)}
        self.sessions = {}              # sid -> {backend: (tokens, ts)}
        self.hits = self.lookups = self.invalidations = 0
        self.budget = {}                # backend -> pool size in tokens (absent/0 = unbounded, TTL only)
        self.churn = {}                 # backend -> cumulative uncached tokens prefilled there

    # ---- anchors ---------------------------------------------------------------------------------------------------
    def _positions(self):
        p = self.stride
        while p <= self.linear_until:
            yield p
            p += self.stride
        p = float(self.linear_until)
        while True:
            p *= self.growth
            yield int(p)

    def anchors(self, pieces, extend=None):
        """pieces: iterable of (text, is_boundary).  Returns Anchors.  `extend`: an Anchors from the same request to continue (reply anchor)."""
        if extend is not None:
            h, total, points = extend.hasher.copy(), extend.total_chars, list(extend.points)
        else:
            h, total, points = hashlib.sha1(), 0, []
        pos_iter = self._positions()
        nxt = next(pos_iter)
        while nxt <= total:
            nxt = next(pos_iter)
        for text, boundary in pieces:
            data = text
            while data:
                room = nxt - total
                if len(data) >= room:
                    seg, data = data[:room], data[room:]
                    h.update(seg.encode("utf-8", "surrogatepass"))
                    total += len(seg)
                    points.append((total, h.copy().digest()[:12]))
                    nxt = next(pos_iter)
                else:
                    h.update(data.encode("utf-8", "surrogatepass"))
                    total += len(data)
                    data = ""
            if boundary and total and (not points or points[-1][0] != total):
                points.append((total, h.copy().digest()[:12]))
        if len(points) > 192:               # bound the work for very long chats: keep the head and the most recent anchors
            points = points[:32] + points[-160:]
        return Anchors(points, total, h)

    def anchors_chat(self, messages, seed=""):
        if seed:
            return self.anchors(itertools.chain([(seed, False)], render_chat(messages)))
        return self.anchors(render_chat(messages))

    def anchors_text(self, prompt):
        return self.anchors([(prompt, True)])

    def extend_with_reply(self, anch, reply_text):
        return self.anchors(render_chat([{"role": "assistant", "content": reply_text}]), extend=anch)

    # ---- table ------------------------------------------------------------------------------------------------------
    def lookup(self, anch, backend, now):
        """-> (cached_tokens, anchor_chars) the backend is believed to hold for this prompt (0, 0 if none)"""
        if not self.enabled or anch is None:
            return 0, 0
        self.lookups += 1
        for chars, dig in reversed(anch.points):
            ent = self.table.get((chars, dig))
            if not ent:
                continue
            tok = ent.get(backend)
            if tok is None:
                continue
            tokens, ts, ch = tok
            if now - ts > self.ttl:
                continue
            b = self.budget.get(backend)
            if b and self.churn.get(backend, 0) - ch > b:
                continue                                  # pushed out of the backend's pool by later prefills
            if tokens < self.min_tokens:
                return 0, 0
            self.hits += 1
            return tokens, chars
        return 0, 0

    def record(self, anch, backend, total_tokens, now, uncached=0):
        """this backend now holds the prompt (total_tokens = the whole prompt, `uncached` of them are prefilled now: pool churn);
        interior anchors get a proportional share"""
        if uncached > 0:
            self.churn[backend] = self.churn.get(backend, 0) + uncached
        if not self.enabled or anch is None or not anch.points:
            return
        total = max(1, anch.total_chars)
        ch = self.churn.get(backend, 0)
        for chars, dig in anch.points:
            tokens = int(total_tokens * chars / total)
            key = (chars, dig)
            ent = self.table.get(key)
            if ent is None:
                ent = self.table[key] = {}
            else:
                self.table.move_to_end(key)
            ent[backend] = (tokens, now, ch)
        while len(self.table) > self.max_entries:
            self.table.popitem(last=False)

    def invalidate(self, anch, backend):
        """the backend's reply showed it does not hold this chain (cached_tokens far below the belief): forget it for that backend"""
        if anch is None:
            return
        self.invalidations += 1
        for key in anch.points:
            ent = self.table.get(key)
            if ent is not None:
                ent.pop(backend, None)
                if not ent:
                    del self.table[key]

    def record_extension(self, ext, prompt_chars, backend, prompt_tokens, completion_tokens, now):
        """anchors of prompt + reply (ext, built with extend_with_reply): record only the points past the prompt"""
        extra = max(1, ext.total_chars - prompt_chars)
        ch = self.churn.get(backend, 0)
        for chars, dig in ext.points:
            if chars <= prompt_chars:
                continue
            tokens = int(prompt_tokens + completion_tokens * (chars - prompt_chars) / extra)
            key = (chars, dig)
            ent = self.table.get(key)
            if ent is None:
                ent = self.table[key] = {}
            else:
                self.table.move_to_end(key)
            ent[backend] = (tokens, now, ch)
        while len(self.table) > self.max_entries:
            self.table.popitem(last=False)

    def forget_backend(self, backend):
        """drop everything believed about a backend (it restarted, or its breaker opened: its cache is gone or unreachable)"""
        for key in list(self.table):
            ent = self.table[key]
            ent.pop(backend, None)
            if not ent:
                del self.table[key]
        for sid in list(self.sessions):
            self.sessions[sid].pop(backend, None)

    def prune(self, now):
        dead = [k for k, ent in self.table.items() if all(now - e[1] > self.ttl for e in ent.values())]
        for k in dead:
            del self.table[k]
        for sid in [s for s, e in self.sessions.items() if all(now - ts > self.ttl for _, ts in e.values())]:
            del self.sessions[sid]

    # ---- sessions -------------------------------------------------------------------------------------------------------
    def lookup_session(self, sid, backend, now, n_in):
        if not self.enabled or not sid:
            return 0
        tok = self.sessions.get(sid, {}).get(backend)
        if not tok or now - tok[1] > self.ttl:
            return 0
        return max(0, min(tok[0], n_in - 1))

    def record_session(self, sid, backend, tokens, now):
        if not self.enabled or not sid:
            return
        self.sessions.setdefault(sid, {})[backend] = (tokens, now)
        if len(self.sessions) > self.max_entries:
            self.sessions.pop(next(iter(self.sessions)))
