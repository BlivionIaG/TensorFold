# Zig HIP port against the Python ROCm engine

What the Python ROCm engine (`src/tensorfold/rocm/`, served by `tensorfold serve --backend rocm`) does, what the Zig
HIP port (`zig/src/families/qwen35/`, served by `tensorfold-native`) does, and what is left. "Zig" is this branch; the
rows say which commit of the parity work closed a gap. Priorities: P0 blocks using the port as a server, P1 is a
behaviour the Python engine has and a user sees, P2 is speed or a corner, P3 is polish.

Output is judged by the invariants (drafted == serial, solo == together, resumed == fresh, tp consistent across runs),
not by bit equality with Python.

## Server and command line

| Feature | Python ROCm | Zig | Gap | Pri | Plan |
| --- | --- | --- | --- | --- | --- |
| `--tp 2/4/8`, `--rank`, `--master`, `--master-port` on `serve` | one process a rank, rank 0 serves HTTP, the others follow | tp only through `tf-qwen35-test lanes --tp`; the server's flag table refuses `--tp`/`--rank`/`--master` | yes | P0 | the native server opens the TCP link and RCCL, rank 0 serves, ranks above 0 run `Worker.follow` |
| `--tp 4`, `--tp 8` | slices and rank-order sums | slices equal Python's at 2, 4 and 8; sums are the rank-order gather; run only at tp 2 | untested at 4 | P0 | commands for a 4-card host in the report; link and flag table accept 4 and 8 |
| window and prompt-cache plan agreed by the ranks | all-reduce min of window and cache | each rank plans alone | yes | P0 | all-gather of the two numbers, min on every rank |
| `--backend rocm` | accepted, `auto` picks it on an AMD GPU | table accepts `auto`/`cuda` on Linux | yes | P0 | accept `rocm` (and `auto`) |
| `--p2p` / `--no-p2p` | sets `NCCL_P2P_DISABLE` before the communicator | none | yes | P1 | same variable, same place |
| `--checkpoint-slots N` | kept prompt entries (default 8, 0 off) | 8, fixed | yes | P1 | flag to `Hip.keepPrompts` |
| `--prompt-cache-gib G` | byte budget (0 off), default the plan's room | the plan's room only | yes | P1 | flag to `Hip.keepPrompts` |
| `--parallel` | `auto` is 1 lane, N lanes | `auto` is 8 lanes (the Mac's) | memory and window differ | P1 | `auto` is 1 on HIP |
| `--context` | default is what the memory fits up to the model's window; an explicit value that does not fit is refused | default capped at 32,768; explicit refused as Python | default | P1 | default is the model's window, fitted by the plan |
| `--no-drafts`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--thinking*`, `--reasoning-effort`, `--thinking-budget`, `--loop-guard`, `--name`, `--alias`, `--api-key*`, `--host`, `--port` | the shared torch server | the shared native server (same table as Metal) | none | | |
| `--mtp-drafts`, `--mtp-confidence` | refused on ROCm | refused | none | | |
| `--kv-dtype`, `--vision*`, `--drafter*`, `--prefill-fp8`, `--precision` | refused on ROCm | refused | none | | |
| `TENSORFOLD_GRAPH=0`, `TF_RCCL_LIB`, `TENSORFOLD_ROCM_SCHEDULE` | read | `TF_HIP_GRAPHS=0`, no RCCL override, `TF_WMMA` | names | P3 | `TENSORFOLD_GRAPH` read as well; `TF_RCCL_LIB` read |
| `--master` as a host name | TCPStore resolves it | the link parses an IPv4 literal | `localhost` and names fail | P3 | resolve through the system resolver |
| Followers waiting for rank 0 | store timeout 600 s | a follower retries the connect for 60 s | shorter | P1 | 600 s |

## Engine

| Feature | Python ROCm | Zig | Gap | Pri | Plan |
| --- | --- | --- | --- | --- | --- |
| Lane rounds, shared forward over every stream's window | yes | yes (window up to 16 rows, copies as well as MTP drafts) | none | | |
| MTP head, up to 3 drafts, side `mtp*.safetensors`, unquantized `fc` | yes | yes; under tp rank 0 holds the whole head and the others skip it | none | | |
| MTP confidence cut | a chain ends after a draft under 0.3 (softmax of its row) | `confidence` is declared and unused; the chain always runs the depth the core asks for and a short chain is an error | yes | P1 | the head also reads each draft's probability; the chain returns the cut length through the backend's `tree` hook |
| Sampling: keyed draw over top_k + MARGIN, top_p, min_p, temperature | yes | the same sampler (lanes/sampling.zig), candidates chosen on the device | none | | |
| Prefix cache: cuts at shared blocks, history and prompt end - 1, eviction order | yes | yes (prefix.zig), mirrored on every rank | none | | |
| Resumed == fresh, solo == together, drafted == serial | yes | yes, checked under tp 2 | none | | |
| tp: sums in rank order past two ranks, one add at two | yes | yes (reduce.zig) | none | | |
| tp: lone stream replays a graph on every rank | yes | graphs are off under tp | yes | P2 | capture the collectives in the graph, rank 0's choice sent with each round |
| Graph replay at tp 1 | a lone stream's windows | any round shape, second sighting | none | | |
| Prompts interleaved with decode (1,024 rows a round while others decode) | yes | a prompt prefills whole before the next round; other streams wait | latency | P2 | needs the core to ask for a prefill step per round (core change, shared with Metal) |
| A cancelled request stops its prefill | the next step | not until the prompt ends | yes | P2 | cancel checked between the engine's 2,048-row spans |
| Start-up warm prefill | `warm()` | none (kernels load on first use) | first request is slower | P3 | `Engine.warm` |
| Memory plan: reserve, scratch, lanes' caches, one kept copy | `memory.plan` | `memory.zig`, the same terms | none | | |
| `response_format`, `guided_*`, `tool_choice` required / named | yes (grammar constraint over the window) | `Info.structures` and `call_gates` are false: the server refuses the request | yes, shared with every native engine | P2 | needs a grammar mask in the lane core; not a HIP change |
| Image input | refused (text only) | refused | none | | |
| W4A16 GPTQ / AWQ | refused at load | refused at load | none | | |

## Models

Both engines read `qwen3_5` and `qwen3_5_moe` MLX affine checkpoints at 2, 3, 4, 5, 6 and 8 bits, groups 32, 64 and
128, widths per tensor from the config, scales and biases as stored.

| Checkpoint | Architecture | Zig run |
| --- | --- | --- |
| `Qwen3.8-27B-MLX-4bit` | `qwen3_5` dense, 64 layers, hidden 5120, 4 KV heads, 1 MTP layer | see results below |
| `Qwen3.8-27B-3bit-mtp-mlx` | the same, 3 bits, embeddings at 4, unquantized MTP `fc`, vision tower entries in the quantization table | see results below |
| `Qwen3.6-35B-A3B-MLX-4bit-MTP` | `qwen3_5_moe`, 256 experts, `mtp-4bit.safetensors` beside | checked before this work |
| `Qwen3.5-0.8B/4B/9B` 3 to 8 bits | `qwen3_5` | 0.8B and 9B checked before this work |

## OpenAI and Anthropic surface

The routes, request fields, errors, streaming and the tool parsers are the native server's, which both Metal and HIP
use; the HIP port adds none and removes none. `POST /v1/decisions` is the Mac's only. The structured-output and
`tool_choice` gaps above are the only API differences that come from the ROCm engine.

## Status after the parity work

| Gap | State |
| --- | --- |
| tp in the server, `--tp/--rank/--master/--master-port/--p2p`, `--backend rocm`, `--checkpoint-slots`, `--prompt-cache-gib`, `--parallel auto`, default window, agreed plan | closed (6c0e968) |
| tp 4 | run through the native server on four V620s: 9B + MTP, 27B 3-bit + MTP and 35B-A3B + MTP, drafted == serial, solo == together |
| MTP confidence cut | closed (dc91a99); the prompt's first drafts run whole, because the core reads no hook there |
| Cancel during prefill | closed (9b23cdf) at tp 1; at tp > 1 a prompt runs to its end |
| `TF_RCCL_LIB`, `TENSORFOLD_GRAPH` | closed (0562c9d, 9b23cdf); `--master` takes an IPv4 literal or `localhost` |
| Graph replay under tp, prompts interleaved with decode, grammar and call gates, start-up warm prefill | open |

Known, not from this work: on the 35B-A3B at tp 4 a prompt of 69 tokens run whole and the same prompt cut at 13 can
draw a different token. The cut run's spans are under 64 rows and take the serial DeltaNet; the whole one takes the
chunked prefill (`TF_GDN_CHUNKED=0` moves the whole run's tokens too), so the two paths differ in the last bits.

## Serving at tp

One process a rank, the same command with its own `--rank`; with every card visible rank r takes card r.

```bash
tensorfold-native serve MODEL --tp 4 --rank 1 --master 127.0.0.1 &
tensorfold-native serve MODEL --tp 4 --rank 2 --master 127.0.0.1 &
tensorfold-native serve MODEL --tp 4 --rank 3 --master 127.0.0.1 &
tensorfold-native serve MODEL --tp 4 --rank 0 --master 127.0.0.1 --name local-model
```

`zig/tests/qwen35/serve_tp.sh DEVS WORLD MODEL_DIR PORT` starts that and checks it with `serve_check.py`.
