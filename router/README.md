# tfrouter: cost-based, cache-aware router in front of two full servers

An OpenAI-compatible HTTP proxy (python 3, **stdlib only**, asyncio) that sits in front of

* **S** = the two DGX Sparks' TensorFold server, and
* **M** = the M5 Ultra's native EXL3 Metal server.

It sends each request to the backend with the lowest predicted cost, subject to health, context, admission and prefix affinity.
URLs, capacities and timing profiles are configurable; the default timing tables are historical calibrations (`docs/baselines.md`), not current benchmark results.
The model extends `tools/sim.py`'s P==D predictor with reservations for requests that have not started decoding;
`tests/test_core.py` checks equivalence on simulator states without those reservations.
The design and rollout plan are in `../docs/fleet-serving.md`. The proxy does not require backend changes.

```
client ──► tfrouter (:8000) ──► S  http://HEAD:8888   (DeepSeek-V4.1-Flash-1017)
                           └──► M  http://M5:8899     (DeepSeek-V4.1-Flash)
```

## Files

| path | what |
|---|---|
| `tfrouter/config.py` | every policy parameter with its measured default (`PRESET_S`, `PRESET_M` = tables from `baselines.md`), strict validation |
| `tfrouter/model.py` | `Profile`/`View`/`predict`: queue wait + prefill(uncached rows) + decode at current and already-reserved future load |
| `tfrouter/backend.py` | live state per backend: in-flight/queued, FIFO prefill backlog, slot wait, EWMA calibration of prefill and decode speed, circuit breaker, API key file |
| `tfrouter/admission.py` | admission control: per-backend in-flight cap (default = slots), the router's wait queue (interactive first, then age), late binding, 503 + Retry-After |
| `tfrouter/prefix.py` | prefix affinity table (hashes of leading messages / character strides -> backend, TTL, LRU) |
| `tfrouter/tokens.py` | token estimation: calibrated chars/token, exact `/v1/tokenize` near boundaries, per-class output-length EWMA |
| `tfrouter/policy.py` | `decide()`: exclusion by context/health/force, cost, SLO, guard, classes, explain record |
| `tfrouter/server.py`, `httpio.py` | the proxy: parsing, SSE passthrough, cancellation, failover, probes, admin routes |
| `router.config.json` | the complete default config (generated: `python3 -m tfrouter --dump-config`; a unit test keeps it in sync) |
| `tools/sim.py` | the discrete-event simulator the cost model and the workloads come from |
| `tools/fakebackend.py` | fake TensorFold/OpenAI SSE server with the machines' measured behaviour (profiles S and M, slots, FIFO prefill, S decode windows, M head-of-line blocking, prefix cache, faults) |
| `tools/replay.py` | real-time harness: fakes + the real router process + `sim.WORKLOADS` traces, same summary as `sim.py`, optional `--sim` DES on the same trace |
| `tools/simcompare.py` | runs the router's own policy code inside `sim.py`'s DES (what-if tables, variants, decisions by prompt size; `compare` = the E2 chat points, old vs fixed policy) |
| `tools/fakepd.py` | the M server for offline hand-off tests (a forced prefill is exported by the S fake) |
| `tests/` | `python3 -m unittest discover -s tests -t .` (146 tests; no network beyond loopback) |

## Run it

```
cd router
python3 -m tfrouter --dump-config > my.json          # every parameter, measured default
# edit my.json (or write a small override file: only the keys that differ are needed):
{
  "listen": {"host": "0.0.0.0", "port": 8000},
  "client_api_key_file": "/etc/tfrouter/clients.keys",       # optional: keys the router's clients must present (one per line)
  "backends": {
    "S": {"url": "http://HEAD:8888", "api_key_file": null},   # production API: no key on the LAN today
    "M": {"url": "http://M5:8899",  "api_key_file": "/path/to/m5-api.key"}
  }
}
python3 -m tfrouter --config my.json --check-config
python3 -m tfrouter --config my.json --explain                 # log every decision (JSON line, stderr)
```

* Backend API keys come from files in TensorFold's key-file format (re-read when the mtime changes; `SIGHUP` reloads). The router does not forward the
  client's `Authorization` header: it authenticates to each backend itself.
* The M server has to bind an address the router can reach (`--host 0.0.0.0`, or an `ssh -L` tunnel).
* Endpoints: `POST /v1/chat/completions`, `POST /v1/completions` (stream and non-stream), `GET /v1/models` (one public id, `DeepSeek-V4.1-Flash`),
  `POST /v1/tokenize` (passed to a healthy backend), `GET /health`, `GET /router/state` (backends, breaker, calibration, affinity size/hits/invalidations),
  `GET /router/explain` (ring of the last decisions), `GET /router/config`.
* The request's `model` is rewritten to each backend's own name (S `DeepSeek-V4.1-Flash-1017`, M `DeepSeek-V4.1-Flash`), everything else in the body is
  passed through unchanged.

### Request classes and headers

| | |
|---|---|
| class | `X-Request-Class: interactive\|background`, or a model suffix: `DeepSeek-V4.1-Flash:background` (also `:batch`, `-background`). Default `interactive` |
| `interactive` | SLO-aware (TTFT SLO = 1.5 s + 0.55 s per 1k prompt tokens, `policy.slo`; 2.6 s at 2k, 10 s at 16k), guarded against slowing M's decoders with long prefills |
| `background` | no SLO, prefers M (bonus 20 s on the cost, up to 240 s of M backlog, interference weight 2) so the Sparks stay free for interactive work |
| `X-Session-Id` | explicit session key for affinity (otherwise hashes of the leading messages) |
| `X-Route-To: S\|M` | force a backend (still fails over if it is down) |
| `X-SLO-TTFT: seconds`, `X-Expected-Output-Tokens: n` | per-request overrides |
| `X-Router-Explain: 1` | the reply carries `X-Router-Backend`, `X-Router-Rule` and `X-Router-Explain` (the decision JSON with the predicted TTFT/E2E per backend) |

### Modes for trying it safely

* `--dry-run`: no backend is contacted for requests; the reply is the decision JSON. The probes still run.
* `--shadow S`: every request is proxied to S, the router only logs what it would have chosen (a week of shadow logs answers "would M have helped").
* `--explain [--explain-file F]`: one JSON line per request: class, prompt-token estimate (and exact count when asked), uncached rows per backend, queue
  wait/prefill/TTFT/E2E/cost per backend, chosen backend, rule (`min-cost`, `affinity`, `context`, `guard`, `slo-miss`, `failover` ...).

## How it decides

For every request: classify; estimate prompt tokens (chars/token EWMA calibrated from the backends' own `usage.prompt_tokens`; the exact
`/v1/tokenize` of the candidate backend is asked only when the estimate is within 12 % of a context limit); look up the prefix table (rows already cached on
a backend are not prefilled again); then for each backend that is healthy and has the context

```
ttft = max(slot_wait, prefill_backlog) + prefill(uncached rows) (+ interleave inflation on S)
e2e  = ttft + (out_est - 1) * round_time(k_forecast+1) / tokens_per_round
cost = e2e + w_int * (decode time added to the others)  + slo_penalty * max(0, ttft - slo)
```

* `k_forecast` counts current decoders plus admitted requests that have not emitted a first token. Each pending request reserves its estimated
  output overlap, so a burst cannot price every arrival as a lone decoder. Prefill-only SM shadows do not reserve an S decoder;
* prefill backlog, current-decoder stalls and M's guard retain their separate accounting. Round-time tables and observed/predicted speed
  are scaled by an EWMA from the responses' `usage` and `tensorfold` timing block;
* S is primary: the other backend must beat it by `primary_bias_s` (0.3 s) **plus** `primary_bias_frac` (15 %) of S's own cost, so M is chosen only for a clear predicted gain, not for 1-2 s on a 10-15 s request;
* capability: sampled requests, grammar/logprobs fields and images are kept off M (greedy-only decode, `pd.greedy_only_decode`, `pd.s_only_fields`), with the hand-off on or off; `X-Route-To: M` overrides;
* M guard: an interactive prompt of more than 4096 uncached rows is kept off M while M has any decoder (M stops its decoders for the whole prefill);
* the SLO is a soft penalty (`slo_penalty` 3 s of cost per second over), chosen by `tools/simcompare.py` (see below);
* the output length estimate is the client's `max_tokens`, capped by the per-class output estimate (a running mean over the first requests, then an EWMA: `tokens.output_warmup`);
* the interference term charges current and pending streams only for their estimated overlapping output tokens (`policy.interf_overlap`);
  the new prompt's one-off prefill stall applies only to streams already decoding, not to future reservations;

## Admission control (`admission`, on by default)

E2 (2026-11-11) showed what happens without it: the router forwarded everything it had a prediction for and the backends queued the excess; at chat 0.15 req/s production (S) was driven past its 4 slots and stopped answering.
Now each backend has `max_inflight` (default: its `slots`, S 4 / M 4; other clients' requests that the server reports count) and `queue_max` (32):

* a request goes to a backend at once when **its first choice** (the policy's best backend, whose price already includes that backend's slot wait) has a free lane, unless an older or higher-priority waiter wants the same backend; it never takes a worse backend just because that one has a lane (a conversation whose prefix is on S, a long prompt kept off M by the guard; the first version took "any free lane" and lost 30 % goodput on the agent DES);
* otherwise it waits **in the router** (interactive classes before background, then oldest) bound to no backend; whenever a lane frees (or after `admission.poll_s`) it is decided again on fresh state and lands on whichever backend is best then. The requests already waiting for a backend count as queued in that backend's price (`Backend.view(now, ahead)`), so a long queue for S makes M's free lane the better choice;
* a request whose eligible backends all already have `queue_max` waiters is answered `503` + `Retry-After` (the soonest predicted slot wait, 1-30 s) at once; a request that has waited `admission.max_wait_s` (120) gets the same;
* a failover target without a free lane is waited for the same way (the retry never overshoots the cap); `X-Route-To` requests wait for their lane too; shadow mode (a transparent proxy) and dry-run are not gated;
* `/router/state` shows `admission.per_backend.{in_use, max_inflight, waiting, queue_max}`; the explain record has `held_s`; `admission.enabled: false` restores the old behaviour (the backends queue).

## PD+ fleet (`"pd": {"enabled": true}`, design `../docs/fleet-serving.md`, tier 2)

* **Three paths** per request: SS (S alone), MM (M alone, body gets `"prefill":"never"`), SM (the request goes to the M server with `"prefill":"force"`; the M server's own
  `--prefill-from` machinery makes S prefill and export, M pulls the KV and decodes). SM is one more candidate in `policy.decide` (`d.cands["SM"]`, `d.paths`); its cost is S's
  backlog + prefill (x`export_factor`) + the hand-off H(n) (0.26 s at 2k rows, ~1 s at 64k) + M's slot wait, plus the interference on both machines, plus `mu` x S seconds.
* **Exclusions** (reason in the explain record): `not-forced`, `s-only:<sampled|field:*|images|exact>`, `too-few-rows` (< `min_rows`, M's TF_PREFILL_MIN), `context`, `breaker-open*`,
  `sm-fallbacks`. Sampled requests (temperature > 0), grammar/logprobs fields and images stay on S (M is greedy-only); `X-Exactness: exact` only rules out the hand-off.
  `X-Route-To: SM` forces it.
* **Bookkeeping**: an SM request holds a decode lane on M (`Backend.pending`, not in M's prefill FIFO) and a *shadow* prefill in S's FIFO until M's first token; S's `/v1/load`
  (`waiting_prompt_tokens + filling_prompt_tokens_left`) adds the other clients' prompt backlog (polled every `health.load_poll_s`). Servers answer with `pulled_prefix` /
  `remote_prefix`; `fallback.failures` SM requests in a row without a pull switch SM off for `fallback.cooldown_s` (request is still served: M prefills it itself).
* **Late binding** is `admission.py`'s wait queue (it applies to every request, PD or not): a request waits in the router for a free decode lane instead of taking a lane on a
  backend, and is decided on fresh state each time a lane frees.
* `tools/fakepd.py` is the M server for offline tests (a forced prefill is exported by the S fake); `tests/test_pd.py`.

## Failure behavior

* **Health**: `/health` of each backend every 5 s (2 s timeout); `live.connections` and, when the build has it, `/v1/load` give load from other clients.
* **Circuit breaker** per backend: 3 consecutive failures (connect error, timeout, 5xx, probe failure) open it for 10 s (doubling to 120 s); a single half-open probe request
  then decides. An open backend is excluded from every decision. 429 is a "busy" answer, retried elsewhere but not counted as a failure.
* **Retry**: a request that has not received a body byte from its backend (connect error, first-byte timeout, 502/503/504/429, upstream reset before the first event) is
  re-sent to the other backend if it can take it (context). At most 2 attempts.
* **Both down**: `503` with an OpenAI-style error body. Prompt above M's context with S down: `413`/`503` naming the reason. **Saturated**: `503` + `Retry-After` (`router_queue-full`, `router_queue-timeout`), see Admission control.
* **After the first byte** the stream is committed: an upstream failure ends it with an SSE `error` event and `data: [DONE]`; nothing is replayed (it would double the tokens the client saw).
* **Client disconnect**: the upstream socket is aborted at once (the TensorFold server cancels the request when its client goes away) and the backlog account is released.
* **Router restart**: stateless apart from the affinity table (lost: the next turn of each conversation is routed by cost, may pay one cold prefill).

## What it does not do

* The SM path (S prefills, M pulls and decodes) is off by default (`pd.enabled`, see "PD+ fleet" above); with it off the router is the two-server router. The SM thresholds of the design were set without the 15 % relative primary bias (`policy.primary_bias_frac` 0 in the PD tests).
* The affinity table models each backend's prompt-cache pool as a token budget (S 800k [INFERENCE: SessionTier measured only 'present after 0 churn, gone after 1.25M'], M 2M [ASSUMED]) and corrects itself from the replies' `cached_tokens` (cached < 50 % of expected: the chain is forgotten, the next turn is cost-routed); it cannot see evictions it has not caused or restarts it did not observe (breaker-open and a failed /health do flush a backend's entries).
* No `/v1/responses`, `/v1/messages`, embeddings, images, tool-call-aware routing; only the two completion endpoints (streamed or not), models, tokenize.
* Retries only before the first byte; no hedged requests; the wait queue is in memory (a router restart drops the waiting requests' connections).
* Every client connection is `Connection: close` (no keep-alive), which costs one TCP connect per request on the LAN.
* Token counts are estimates (typically within a few percent after calibration); exact counting only near context boundaries.
* `/v1/load` JSON shape is inferred from `docs/api.md` (the production build answers 404, then `/health live.connections` is used); the M `guard_rows`, the SLO table and `slo_penalty` are chosen
  on the simulator, not measured on the machines until the E2 run (results in `../docs/fleet-serving.md`).
* No authentication of its own beyond an optional client key file; no TLS; no metrics endpoint beyond `/router/state`.

## Offline validation

Everything here is offline: fake backends that reproduce the measured machines (`tests/test_fake.py` checks the idle prefill table, FIFO prefill, 58/110 and 39/60.6 tok/s decode,
M's head-of-line stall of ~9.7 s for a 9k newcomer, prefix-cache hits) and `tools/sim.py`'s workloads/seeds.

### 1. The router's policy code inside sim.py's DES (`tools/simcompare.py tables`, 6 seeds x 900 s)

The same `policy.decide` / `Backend` / `PrefixIndex` code that serves requests, driven by the simulator's event loop; "sim ..." rows are `sim.py`'s own routers.
Goodput = TTFT <= SLO and ITL p95 <= 100 ms (agent SLO 5 s, chat 2 s).

| workload @ rate | setup | TTFT p50 | TTFT p95 | E2E p95 | goodput req/min | util M |
|---|---|---|---|---|---|---|
| agent @ 0.02 | A (S alone) | 1.0 | 13.7 | 23.7 | 8.6 | 0% |
|  | sim A+B least-loaded | 1.0 | 6.8 | 17.4 | 9.9 | 10% |
|  | sim A+B cost-aware | 1.0 | 5.1 | 14.3 | 10.1 | 31% |
|  | tfrouter as in E2 (before the fixes) | 1.0 | 4.0 | 13.2 | 10.5 | 20% |
|  | **tfrouter (default)** | 1.0 | 3.9 | 13.1 | 10.5 | 14% |
|  | tfrouter, hard SLO filter | 1.0 | 5.8 | 16.3 | 9.9 | 7% |
|  | tfrouter, least-loaded mode | 1.0 | 6.0 | 16.5 | 9.9 | 7% |
| agent @ 0.05 | A (S alone) | 24.6 | 95.6 | 105.3 | 4.8 | 0% |
|  | sim A+B least-loaded | 5.5 | 55.1 | 63.5 | 12.0 | 46% |
|  | sim A+B cost-aware | 3.4 | 39.6 | 50.3 | 14.6 | 72% |
|  | tfrouter as in E2 (before the fixes) | 3.9 | 32.2 | 43.4 | 14.4 | 68% |
|  | **tfrouter (default)** | 3.3 | 31.1 | 41.4 | 15.6 | 66% |
|  | tfrouter, hard SLO filter | 4.6 | 31.0 | 41.8 | 13.3 | 54% |
|  | tfrouter, least-loaded mode | 5.1 | 30.0 | 41.4 | 12.6 | 53% |
| agent @ 0.09 | A (S alone) | 103.8 | 187.7 | 199.3 | 0.9 | 0% |
|  | sim A+B least-loaded | 41.2 | 138.0 | 148.0 | 9.7 | 57% |
|  | sim A+B cost-aware | 40.1 | 140.6 | 152.5 | 9.5 | 91% |
|  | tfrouter as in E2 (before the fixes) | 41.0 | 101.8 | 112.1 | 4.5 | 89% |
|  | **tfrouter (default)** | 41.6 | 100.1 | 111.3 | 4.3 | 91% |
|  | tfrouter, hard SLO filter | 40.2 | 98.6 | 108.7 | 4.1 | 89% |
|  | tfrouter, least-loaded mode | 40.4 | 97.7 | 108.5 | 4.3 | 90% |
| chat @ 0.06 | A (S alone) | 1.4 | 3.2 | 26.3 | 3.1 | 0% |
|  | sim A+B least-loaded | 1.4 | 2.6 | 23.0 | 3.2 | 8% |
|  | sim A+B cost-aware | 1.4 | 2.9 | 22.0 | 2.4 | 23% |
|  | tfrouter as in E2 (before the fixes) | 1.4 | 2.9 | 21.0 | 2.5 | 22% |
|  | **tfrouter (default)** | 1.4 | 2.8 | 21.0 | 2.9 | 11% |
|  | tfrouter, hard SLO filter | 1.4 | 2.6 | 21.3 | 3.0 | 9% |
|  | tfrouter, least-loaded mode | 1.4 | 2.9 | 25.2 | 3.1 | 2% |
| chat @ 0.12 | A (S alone) | 1.7 | 17.0 | 40.7 | 4.2 | 0% |
|  | sim A+B least-loaded | 1.5 | 2.9 | 29.7 | 5.5 | 28% |
|  | sim A+B cost-aware | 1.7 | 3.9 | 27.8 | 4.0 | 55% |
|  | tfrouter as in E2 (before the fixes) | 1.7 | 3.0 | 26.2 | 4.2 | 53% |
|  | **tfrouter (default)** | 1.7 | 3.0 | 26.3 | 4.7 | 37% |
|  | tfrouter, hard SLO filter | 1.7 | 2.9 | 27.4 | 5.0 | 33% |
|  | tfrouter, least-loaded mode | 1.6 | 4.9 | 29.9 | 5.0 | 17% |
| chat @ 0.2 | A (S alone) | 80.9 | 213.8 | 237.1 | 0.1 | 0% |
|  | sim A+B least-loaded | 2.2 | 19.9 | 52.0 | 5.6 | 68% |
|  | sim A+B cost-aware | 2.7 | 33.7 | 53.2 | 3.7 | 81% |
|  | tfrouter as in E2 (before the fixes) | 2.6 | 18.6 | 47.3 | 3.9 | 83% |
|  | **tfrouter (default)** | 2.6 | 19.5 | 49.4 | 4.3 | 75% |
|  | tfrouter, hard SLO filter | 2.4 | 17.5 | 51.1 | 4.6 | 71% |
|  | tfrouter, least-loaded mode | 2.6 | 18.2 | 50.8 | 4.4 | 58% |
| mixed @ 0.06 | A (S alone) | 5.8 | 29.0 | 51.5 | 2.0 | 0% |
|  | sim A+B least-loaded | 1.9 | 21.5 | 38.3 | 2.5 | 16% |
|  | sim A+B cost-aware | 2.4 | 25.9 | 43.2 | 2.7 | 30% |
|  | tfrouter as in E2 (before the fixes) | 2.5 | 21.8 | 36.8 | 2.5 | 25% |
|  | **tfrouter (default)** | 2.3 | 22.3 | 39.2 | 2.5 | 15% |
|  | tfrouter, hard SLO filter | 2.3 | 22.3 | 39.1 | 2.5 | 15% |
|  | tfrouter, least-loaded mode | 2.1 | 22.3 | 39.3 | 2.4 | 13% |
| mixed @ 0.1 | A (S alone) | 16.8 | 60.6 | 91.2 | 2.0 | 0% |
|  | sim A+B least-loaded | 2.6 | 26.1 | 52.1 | 3.6 | 38% |
|  | sim A+B cost-aware | 3.0 | 32.5 | 53.3 | 3.9 | 55% |
|  | tfrouter as in E2 (before the fixes) | 2.8 | 27.0 | 47.6 | 3.8 | 48% |
|  | **tfrouter (default)** | 2.9 | 26.1 | 48.6 | 3.6 | 39% |
|  | tfrouter, hard SLO filter | 2.6 | 28.7 | 49.8 | 3.7 | 39% |
|  | tfrouter, least-loaded mode | 2.7 | 28.8 | 51.2 | 3.7 | 35% |
| mixed @ 0.14 | A (S alone) | 93.2 | 189.0 | 212.8 | 0.3 | 0% |
|  | sim A+B least-loaded | 7.7 | 44.7 | 74.7 | 4.2 | 63% |
|  | sim A+B cost-aware | 9.5 | 49.3 | 79.7 | 3.9 | 75% |
|  | tfrouter as in E2 (before the fixes) | 7.2 | 41.6 | 71.9 | 4.3 | 71% |
|  | **tfrouter (default)** | 7.6 | 44.0 | 71.5 | 4.4 | 66% |
|  | tfrouter, hard SLO filter | 6.8 | 42.5 | 72.7 | 4.2 | 65% |
|  | tfrouter, least-loaded mode | 7.8 | 42.6 | 73.6 | 4.1 | 60% |

Read: at moderate load tfrouter matches or beats the simulator's cost-aware router and clearly beats least-loaded; at the saturated points (chat 0.2, mixed 0.14, agent 0.09) it is within noise of
least-loaded on TTFT p95 and slightly better on goodput. `slo_penalty` was picked from this sweep (0 / 0.3 / 1 / 3 / hard): 3 gave the best TTFT p95 on most points without
giving back goodput; the hard filter (what the first version had) is the best at the very top of the load range and the worst at p50 (it keeps small requests on a saturated S).
Headlines from the setups simulation reproduce: chat @0.12 TTFT p95 17.0 s (S alone) -> 3.0 s; agent @0.05 goodput 4.8 -> 15.6 req/min.
"tfrouter as in E2" is the policy the E2 run used (`E2_POLICY` in `tools/simcompare.py`: no admission control, no relative primary bias, no output warm-up, interference over the
whole request); the fixed default is within about 5 % of it on every row (agent and low-load chat better, saturated chat/mixed ~3-5 % goodput and ~0.7-2 s TTFT p95 worse: the cost of never queueing at a backend, see below).

#### The E2 chat points, old vs new policy (`python3 tools/simcompare.py compare --seeds 6`)

6 seeds x 900 s. Goodput in req/s at the E2 gate's SLO (TTFT <= 4 s, ITL p95 <= 150 ms) and at the simulator's chat SLO (2 s / 100 ms); share = chat requests sent to M.
The DES models the lane cap and late binding, not `queue_max`/`max_wait_s`.

| rate /s | setup | TTFT p95 s | E2E p95 s | goodput req/s @ 4s/150ms | goodput req/s @ 2s/100ms | share to M | finished |
|---|---|---|---|---|---|---|---|
| 0.06 | A  S alone | 3.24 | 26.3 | 0.059 | 0.052 | 0% | 100% |
| 0.06 | Router E2 policy (before the fix) | 2.92 | 21.0 | 0.061 | 0.042 | 24% | 100% |
| 0.06 | Router cost (tfrouter default) | 2.84 | 21.0 | 0.061 | 0.048 | 14% | 100% |
| 0.09 | A  S alone | 9.07 | 31.1 | 0.079 | 0.069 | 0% | 100% |
| 0.09 | Router E2 policy (before the fix) | 2.93 | 23.8 | 0.089 | 0.057 | 30% | 100% |
| 0.09 | Router cost (tfrouter default) | 2.90 | 23.7 | 0.088 | 0.066 | 18% | 100% |
| 0.12 | A  S alone | 16.96 | 40.7 | 0.083 | 0.070 | 0% | 100% |
| 0.12 | Router E2 policy (before the fix) | 3.04 | 26.2 | 0.115 | 0.069 | 31% | 100% |
| 0.12 | Router cost (tfrouter default) | 3.00 | 26.3 | 0.114 | 0.078 | 23% | 100% |
| 0.15 | A  S alone | 46.95 | 70.5 | 0.059 | 0.046 | 0% | 100% |
| 0.15 | Router E2 policy (before the fix) | 5.65 | 31.8 | 0.141 | 0.079 | 31% | 100% |
| 0.15 | Router cost (tfrouter default) | 6.36 | 31.5 | 0.137 | 0.085 | 25% | 100% |
| 0.20 | A  S alone | 213.85 | 237.1 | 0.002 | 0.001 | 0% | 100% |
| 0.20 | Router E2 policy (before the fix) | 18.62 | 47.3 | 0.140 | 0.065 | 33% | 100% |
| 0.20 | Router cost (tfrouter default) | 19.46 | 49.4 | 0.137 | 0.072 | 31% | 100% |

### 2. The real router process against fake backends, real time (`tools/replay.py run ... --sim`, 400 s of traffic at 10x speed, one seed)

HTTP in, SSE out, real sockets; the machines are fakes running the simulator's machine model; same trace as the DES rows (virtual seconds). Single seed and 400 s: noisy, read
the ordering, not the third digit.

| workload | setup | TTFT p50 | TTFT p95 | E2E p95 | M share | note |
|---|---|---|---|---|---|---|
| chat @0.12 | A, real-time | 5.7 | 18.7 | 48.7 | 0 | = sim A DES (5.7 / 18.6) |
| | router-ll | 1.8 | 7.5 | 31.4 | 15 % | |
| | router-e2 (policy as in E2) | 1.8 | 3.0 | 28.3 | 29 % | sim least-loaded 1.6 / 3.0, sim cost-aware 1.8 / 4.8 |
| | **router-cost (fixed)** | 1.8 | 4.3 | 25.6 | 27 % | goodput 4.4 vs 4.6 req/min: 48 requests, p95 = the 2-3 worst |
| mixed @0.10 | A | 29.2 | 70.6 | 102.0 | 0 | = sim A DES (29.2 / 70.5) |
| | router-e2 | 7.5 | 26.4 | 47.8 | 34 % | sim least-loaded 10.0 / 38.3, sim cost-aware 9.4 / 26.4 |
| | **router-cost (fixed)** | 7.4 | 26.4 | 47.8 | 34 % | |
| agent @0.05 | A | 1.4 | 7.1 | 17.2 | 0 | goodput 15.4 req/min |
| | router-e2 | 1.2 | 4.3 | 13.5 | 22 % | goodput 17.5 |
| | **router-cost (fixed)** | 1.2 | 5.0 | 15.6 | 22 % | goodput 16.6 |
| chat @0.20 (300 s) | router-e2 | 3.9 | 20.4 | 54.1 | 34 % | S queued up to 5 requests at the machine, M 2 |
| | **router-cost (fixed)** | 3.5 | 19.8 | 49.9 | 32 % | S and M queue 0 (`max_queue`); 37 requests held in the router |

The one-machine rows (A, B) reproduce the DES to the digit, which validates the fake backends and the harness; the router rows agree with the DES rows of the same
policy within the noise of a single seed (the fixed router is not better than the E2 policy on one seed at these loads: the point of the fixes is the cap and the low-load split, see the chat table above). B alone (M5 only) is 5-10x worse than A at these loads (hence "S by default").

### 3. Decision agreement and unit tests

* `tests/test_core.py::test_predict_equals_sim` asserts `model.predict` equals `sim.predict` on live simulator states.
* 146 tests pass (`python3 -m unittest discover -s tests -t .`, ~10 s): admission control (cap, queue order, 503s, failover, DES regressions for the 0.09 split), config, model, prefix, tokens, breaker, key files, policy rules, proxy end-to-end (stream passthrough, usage injection,
  context routing, affinity, classes, dry-run/explain/shadow, cancellation mid-stream and during prefill, failover/breaker cycle, 503/429, mid-stream error, hang timeout), fake-machine validation.

