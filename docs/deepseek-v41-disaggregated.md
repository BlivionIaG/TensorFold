# DeepSeek-V4.1 disaggregated prefill and decode (GB10 pair + Apple Silicon)

A pair of NVIDIA GB10 machines (two DGX Spark or ASUS Ascent GX10, tensor-parallel over CUDA) **prefill** the prompt; an
Apple Silicon Mac (Metal, EXL3) **decodes** it and serves the API. The prefill state crosses on a ConnectX-5 cable through
[MCDMA](https://github.com/ashhart/MCDMA) (RDMA from macOS), or over plain HTTP where there is no such cable.

This page is the operator recipe. The wire format, the library and the exactness contract are in
[deepseek-v41-handoff.md](deepseek-v41-handoff.md); the model lanes are in [deepseek-v41.md](deepseek-v41.md).

> Status: the path works end to end (link, hand-off, landing, fallback) but, with the engines as they are, it does **not**
> lower time to first token: see [Measured](#11-measured). Use it for capacity, or when the pair prefills faster than the Mac.

> A hand-off prefix is outside the exactness contract: it is opt-in (`--prefill-from`), labelled `remote_prefix` in the
> reply, and decode after it is not bit-identical to a Mac-only prefill. Everything below keeps the Mac able to prefill
> by itself: a producer that is down, slow or wrong only costs the Mac its normal prefill.

## What runs where

```
   client ──HTTP──▶  Mac (Metal): `tensorfold-native serve ... --prefill-from http://PRODUCER:PORT --mcdma-link NAME`
                       │  1. trigger the export     POST /v1/completions + kv_transfer_params   (HTTP, LAN)
                       │  2. pull the state         MKVH protocol 2 over the link NAME          (RDMA, CX5 cable)
                       │  3. land it, run the last prompt token, decode
                       ▼
   GB10 pair (CUDA, TP=2): rank 0 `tensorfold-dsv41 serve ... --mcdma-produce NAME`, rank 1 `tf-dsv41 follow`
                       prefills the prompt to its last-but-one token (all but the Mac's tail), exports rank 0's state, holds it until pulled
```

Only the payload (about 3.7 KB per prompt token, 31 MiB for 8,704 tokens) uses the cable. The link's set-up datagrams and
the trigger use the LAN, so the Mac needs both.

## Requirements

| part | requirement |
|---|---|
| Mac | Apple Silicon with enough memory for the EXL3 pack (the 2.9 bpw pack needs about 215 GB resident), macOS 27, `rdma_ctl enable`, the MCDMA native driver installed, a ConnectX-5 in a Thunderbolt 5 enclosure |
| GB10 pair | two machines with the CUDA stack and the inter-machine RoCE link of the TP pair (this guide calls its devices the *pair devices*); one more ConnectX-7 port free for the Mac, and a free RDMA GID |
| cables | 100G passive QSFP28 whose EEPROM the ConnectX-7 accepts (see [Cables](#cables)) |
| LAN | both ends on one layer-2 segment (the exchange uses link-local datagrams with hop limit 255) |
| model | the same checkpoint files on both sides: the manifest names the checkpoint by the hash of `config.json` and `model.safetensors.index.json` |

## 1. The Mac

Follow MCDMA's `docs/install.md` (Recovery: reduced security, user-managed kernel extensions, `rdma_ctl enable`; install
the kext with the enclosure disconnected, approve it, reboot with the enclosure attached). Then:

```
kmutil showloaded --list-only | grep org.mcdma.cx5.native      # loaded
ibv_devices                                                    # rdma_mcrdma0, rdma_mcrdma1
ibv_devinfo -d rdma_mcrdma1 | grep state                       # PORT_ACTIVE once a cable links
```

## 2. Cables

The ConnectX-7 identifies a QSFP module from byte 0 of its EEPROM (`ethtool -m IF`: identifier `0x11`). A cable whose
EEPROM does not name a QSFP28 leaves the port without carrier: `mstlink -d PCI` reports status opcode 1036 ("Module's PMD
type is not enabled") and the port never links at any speed. Use a cable the card recognises (MCDMA's guide tested the
Mellanox MCP1600-C001E30N; the Spark user guide lists others). Check the port before going further:

```
ethtool IF | grep -E 'Link detected|Speed'                     # yes, 100000Mb/s
sudo mstlink -d PCI | grep -E 'State|Opcode'                   # Active, opcode 0
```

The two Mac ports may be crossed at the enclosure: if a byte-verified transfer fails with `CQ status=12` (retry exceeded),
swap which Mac interface you pair with which machine (step 4 finds the right one).

## 3. Each pair machine: the Mac-facing port

Leave the pair's own RoCE port and its settings alone. On the *other* port (here `PEER_IF`, RDMA device `PEER_DEV`):

```
sudo nmcli dev set PEER_IF managed no          # until reboot; stops NetworkManager replacing the MTU and addresses
sudo ip link set dev PEER_IF mtu 9000 up
cat /sys/class/net/PEER_IF/address             # PEER_MAC
for i in 0 1 2 3; do echo $i $(cat /sys/class/infiniband/PEER_DEV/ports/1/gids/$i) \
                          $(cat /sys/class/infiniband/PEER_DEV/ports/1/gid_attrs/types/$i); done
```

Use the RoCE v2 GID that is the MAC-derived link-local address (`fe80::` + the EUI-64 of `PEER_MAC`); its index is
`PEER_GID_INDEX`.

**Pin NCCL to the pair's own devices on every launch of the TP pair** (`NCCL_IB_HCA==PAIR_DEV_A,PAIR_DEV_B`, the exact-match
form). Once the Mac-facing port has carrier NCCL's autodetection adds it to the TP rings; a ring step then goes to the
wrong machine, and the pair hangs at start-up at 100 % CPU with `IBV_WC_RETRY_EXC_ERR` in NCCL's log.

## 4. Address the link and prove it

On each Mac interface that has a cable (`mcrdmaN`, MAC `MAC_HWADDR`) and its peer (`PEER_GID`, `PEER_MAC`):

```
# Mac, from the MCDMA checkout
sudo python3 tools/restore-rdma.py --interface mcrdmaN --expected-mac MAC_HWADDR --peer-gid PEER_GID --peer-mac PEER_MAC
# peer: the reciprocal static neighbour (the Mac's address-only interface answers no neighbour discovery)
sudo ip -6 neigh replace MAC_GID lladdr MAC_HWADDR nud permanent dev PEER_IF
```

`MAC_GID` is the Mac's MAC-derived link-local address. Both entries are lost on reboot: put them in a boot script on each
side. Then run MCDMA's four-way byte check (`tools/native_cross_host.py`, see its `docs/install.md` step 5) for each pairing
and keep the pairing that passes; require WRITE and READ in both directions, bytes verified, in all three modes.

## 5. The link daemons

`mcdma-rpcd` carries the request/reply mailbox. Its set-up exchange runs on a shared **LAN** segment, not on the cable: the
`VIA` of each end is `LAN_IF/fe80::OTHER_ENDS_LAN_LINK_LOCAL`. (Naming the cable's interface fails with
`connect failed with status 6`.)

```
# producer machine (rank 0's host), as the user that runs the server, started BEFORE the server
mcdma-rpcd listen NAME PEER_DEV PEER_GID_INDEX 4096 LAN_IF/fe80::MAC_LAN_LINK_LOCAL:18620 4 64

# Mac, from a Terminal or an ssh session (a command-line tool there may use the LAN; a process detached from it gets
# HostUnreachable). At boot, run it as the launchd job of section 9.
mcdma-rpcd connect NAME,en0/fe80::PRODUCER_LAN_LINK_LOCAL,18620,rdma_mcrdmaN,0,4096,4,64
```

The listen daemon logs `peer ready over roce` once the Mac connects. `NAME` is the link name used below.

## 6. The prefill pair

Rank 1 is `tf-dsv41 follow`; rank 0 is `tensorfold-dsv41 serve` with one more flag. The pair needs a profile whose
prefill state can be exported: a paged pool. Replay prefill (`TF_DSV41_PREFILL=replay`: the cut stands after the decoder
replay, see below) and prompts in pieces (`TF_DSV41_PREFILL_PIECES=1`) are exported: a pair that serves its own lanes with
pieces exports in the same process, and the decode lanes keep running while a prompt's pieces and its export run. A split pool
(`TF_DSV41_KV_SPLIT=1`) is exported by shard, which is how a pair holds a prompt of up to a million tokens: rank 0 exports
shard 0 with `--mcdma-produce NAME`, rank 1 (`tf-dsv41 follow`) exports its own pages on a listen link of its own host named by
`TF_DSV41_PRODUCE_LINK=NAME2`, and the Mac pulls both at once with `--mcdma-link NAME,NAME2`. `TF_DSV41_POOL_TOKENS` must hold
the prompt: a split pool needs about 1,060 B a token on each rank (a replicated one 1,790).

```
# both machines
export NCCL_IB_HCA==PAIR_DEV_A,PAIR_DEV_B TF_DSV41_FORMS_INPLACE=1 TF_DSV41_SAMPLING=1 TF_DSV41_POOL_TOKENS=65536
export TF_DSV41_GM_V2=1 TF_DSV41_GM_V3=1 TF_DSV41_PF_DENSE=fused TF_DSV41_PF_DENSE_TABLE=$TABLE TF_DSV41_PF_OVERLAP=4
export TF_DSV41_L2PF=1 TF_DSV41_L2PF_MB=12 TF_DSV41_L2PF_PACE_GBPS=150 TF_DSV41_PREFETCH_AHEAD=1 TF_DSV41_ENGRAM_AIO=1
export TF_DSV41_BRANCHES=1 TF_DSV41_BRANCHES_PRIO=side
# rank 0 only
tensorfold-dsv41 serve $MODEL --host 0.0.0.0 --port 8000 --name DeepSeek-V4.1-Flash-TF --no-thinking \
    --mcdma-produce NAME --api-key-file $KEYFILE
```

Do **not** set `TF_DSV41_HANDOFF_DUMP` in service: it writes each export to disk on the engine thread.

Rank 0 logs `mcdma handoff: attached to the listen daemon on link NAME` and `exporting DeepSeek-V4.1 states as MKVH protocol 2
on link NAME`; a start refusal names the knob.

### How fast the pair prefills, and why not faster

The export-compatible profile above is **not** the pair's fastest. Measured on two GB10 (TP=2), tokens per second, best of
three, prompt length 1k / 8k / 32k:

| profile | 1k | 8k | 32k | exportable |
|---|---|---|---|---|
| serving profile (replay prefill, 4,096-row pieces, split KV) | 1.60k | 2.16k | 2.53k | no |
| replay prefill, 4,096 rows, replicated KV, no pieces | 1.64k | 2.17k | 2.59k | yes |
| export profile, no tuning | 0.98k | 1.18k | 1.19k | yes |
| export profile with the knobs above | 1.04k | 1.19k | 1.23k | yes, byte-identical to the untuned export |

The knobs above add 0.5 to 6 %: they are optional. What makes the serving profile fast is replay prefill
(`TF_DSV41_PF_4K`, `TF_DSV41_PF_TBO` and `TF_DSV41_PREFILL_ROWS` above 2048 are refused without it). By default
(`TF_DSV41_REPLAY_EXACT`) the decoder layers replay over their exact cone at the cut: the encoder pass keeps the last
2,541 rows of the first decoder layer's input, and the decoder pass runs in programs of `TF_DSV41_REPLAY_GROUP` layers
(default 4, rounded up to index sources), each over its layers' own cone, so those layers' rings and the drafter's taps
are expected to equal a full prefill's up to the CUDA kernels' reduction order [INFERENCE, not yet compared on the pair]. `TF_DSV41_REPLAY_EXACT=0` keeps the bounded replay over the
last 127 rows (the window's oldest row is zero in the export; normalized RMS 0.3 to 0.5 against a full prefill, growing
with layer depth); `TF_DSV41_PF_TBO=1` also uses it. By default (`TF_DSV41_REPLAY_WARM=1`) a prompt entry is saved after the exact replay up to its position, so its decoder layers' SWA rings are a full prefill's and ride in the entry; a follow-up prompt resumes it with any number of new rows, and the replay at its end runs over the rows since the entry only (its first row reads the restored rings). Level 1 then puts the cone ring back from a host copy, so the prompt's own replay runs as without the entry (bit-identical cold prompts) at the price of a second cone replay for each prompt that saves an entry; level 2 skips that and runs the prompt's own replay over the rows since the entry (cheaper, not bitwise a cold prompt's). `TF_DSV41_REPLAY_WARM=0` keeps the old rule: a saved session that would leave fewer than the cone's rows after it (at most 2,541) is not resumed under the exact replay and the prompt prefills from 0. A drafter's context after a short warm resume starts at the entry (its taps cover the rows since it), so its first drafts see fewer rows. The sliding-window rows
below layer 20, the carries, the compressed pages and the index keys are byte-identical either way. The exact replay runs
about 30.5k layer-rows against 2.5k for the bounded one [INFERENCE, not yet measured on the pair]. Prefill on the plain export profile costs about 0.83 ms a row at every length from 1k to 32k: it is GPU-compute bound, not Engram bound (7 %).
A prompt of 61k tokens (the pool's 65,536-token cap) exports in 51 s.
Of the knobs tried, `TF_DSV41_ENGRAM_GATE`, `TF_DSV41_ENGRAM_PREFETCH`, `TF_DSV41_INDEX_BUDGET_MIB`,
`TF_DSV41_ROUTER_GROUP_ROT` and `TF_DSV41_KV_NORM_STORE` together made the export fail (HTTP 409, "the prompt pass ended
before its state was saved") and bought nothing; which one is the cause was not bisected.

The exact replay stays on the prompt's one stream; `TF_DSV41_REPLAY_OVERLAP` (default 1) only moves the host's wait.
A second stream beside the next prompt's encoder segment shares the forward's window, layer, expert and router scratch (72
roles in the forward's buffers, 74 in the 4K workspace, from the segment's first call): event ordering alone overlaps
nothing. A replay namespace of its own (host test `replay_overlap_test`) leaves only the five cone-ring roles shared, ordered
by one event before the segment's last call, but costs 1.2 GiB a rank at 2,541 rows beside the workspace's, and the roles
are not the only shared scratch: the gm_plan mask, the long-prompt scratch and the exchange buffers are single buffers, and
two streams' collectives on one communicator cannot be shown deadlock-free off the GPU. It also buys nothing in the
serving flow: the round issues pieces before finals, and the hand-off ingest or the last token's window waits for the
replay before the next prompt's piece can be issued. So with the knob on, a replay in the forward's buffers ends on an
event pair and the host waits for it where it reads its rows (the DSpark ingest, the export cut); a replay in the workspace
or the lazy index scratch still waits at its end, to free them. The result is the same bytes. The `replay cone:` line prints
the replay's GPU ms, the ms it was issued in, the ms the host waited and the ms overlapped; compare `=0` and `=1` over 1k to
32k prompts in the replay profile. [INFERENCE: the gain is the host's issue bubble only, not yet measured on the pair.]

The end-to-end comparison against a Mac that prefills at about 0.9 to 1.0k tokens a second is in [Measured](#11-measured).
Freeing the Mac's GPU from long prompts while it decodes other requests is the other expected gain; it is not measured here.

## 7. The Mac decoder

```
TF_PREFILL_API_KEY_FILE=$KEYFILE tensorfold-native serve $PACK --engram-stream-dir $ENGRAM \
    --chat-template $PACK/chat_template.jinja --context 32768 --prompt-cache-gib 4 \
    --prefill-from http://PRODUCER:8000 --mcdma-link NAME
```

`--prefill-from` is the producer's HTTP address (the trigger); `--mcdma-link NAME` is the daemon link (the payload).
`TF_DSV41_PREFILL_STEP=4096` is the Mac's own prefill step. A prompt shorter than `TF_PREFILL_MIN` is prefilled on the
Mac.

The Mac runs only the prompt's last token after the pull (`TF_PREFILL_TAIL`, default 1: a one-row, decode-class step, so no
prefill pass runs on the Mac), and the pair prefills all the rest, tail included. `TF_PREFILL_TAIL=0` restores the 512-token
boundary (`TF_PREFILL_CHUNK`), which leaves the Mac a 96 to 461 token pass of 0.6 to 1.1 s. Not measured on the pair yet; see
[the hand-off page](deepseek-v41-handoff.md#how-much-of-the-prompt-is-handed-over) for what moves, what is not shipped and the drafter's start.

Leave `--keep-warm` at its default (900 s) and do not pass `--keep-warm 0`: while the pair prefills, the Mac's GPU is idle for
seconds, macOS un-wires an idle model after one to two seconds, and the first pass after the hand-off then waits 0.9 to 2.3 s
to re-wire it. The keepalive ticks while a hand-off is awaited. A quiet server (a request after more than a few seconds of
idleness) can still pay that re-wire once.

### Several requests, and when a prompt goes to the pair at all

The pair prefills one prompt at a time, and the Mac holds one hand-off at a time. `TF_PREFILL_ROUTE=auto` (the default)
decides each prompt that reaches `TF_PREFILL_MIN` when it is queued, and again while it waits for a lane:

* **The pair is busy with another prompt**: the prompt is prefilled on the Mac at once, as a short prompt is, unless the pull
  under way ends so soon that waiting for it and then pulling still beats the Mac's own prefill. It is not queued behind the
  pull with the Mac idle (a mutex held across the whole pull did that).
* **A stream is decoding on the Mac**: the prompt goes to the pair even if the pair is no faster, because a prompt pass on
  the Mac would stall that stream for seconds (inter-token latency isolation), unless the pair is busy as above.
* **Otherwise** it goes to the pair only if the pull is predicted no slower than the Mac's own prefill: the prefix at the
  pair's measured rate plus the Mac's last-token pass, against the rest of the prompt at the Mac's measured rate. The rates
  are running averages of the pulls and prompt passes this server timed (a pass under 1024 rows says too little to count;
  one sample moves a rate by a third to three times at most). Until it has timed its own it assumes 1500 tokens a second for
  the Mac (`TF_PREFILL_LOCAL_TPS`), 2000 for a pull (`TF_PREFILL_REMOTE_TPS`) and 0.6 s for the last-token pass, the export
  profile's numbers: with them a lone prompt of under about 19k tokens stays on the Mac, whose prefill is as fast as the
  pair's there (the pair's gain is the Mac's freedom, not the first token). The pair's rate is learned only from pulls, so
  a pair that is faster than that (a replay export) is first used when a stream decodes or the prompt is long; set
  `TF_PREFILL_REMOTE_TPS` to what it does to use it from the start.

The server logs `remote prefill: a prompt of N tokens (M cached) is prefilled here: REASON` once for each reason a prompt
is kept off the pair for now. `TF_PREFILL_ROUTE=always` restores the old behaviour: every prompt past `TF_PREFILL_MIN` goes
to the pair, queued behind any pull under way, however long that is. Not yet measured on the machines: two concurrent 8k
prompts, and the rates the average settles at.

A request can overrule the weighing with a `prefill` field in its body (`/v1/chat/completions` and `/v1/completions`): `"force"`
sends it to the pair whatever the load or the predicted times say and waits for a pull under way instead of prefilling here,
`"never"` prefills it here, `"auto"` (the default) is the decision above. Both words keep the rules that need no judgement: a
prompt too short to pull, a pair that is off or not answering, and a prefix the cache already covers are still prefilled here.
The reply's `remote_prefix` says whether the pair prefilled it. A router in front of the Mac and the Sparks uses the field to
send a prompt through the Mac to the pair (`force`) or keep it local (`never`) from its own view of both machines.
### A pair that goes away

* **Deadline.** A pull is bound by `TF_PREFILL_TIMEOUT_S` if set, else by 15 s plus three times the pull the Mac's own
  timings predict for it (the prefix at the pair's measured rate plus the last-token pass), between 30 s and 30 min; the
  trigger and, over HTTP, the transfer each get that. A pair that stopped answering is found in about a minute, not after
  an hour-long default.
* **Circuit breaker.** After `TF_PREFILL_BREAKER` failed pulls in a row (default 3; the link down, a connect or trigger
  that timed out; a refusal or a mismatch is an answer and does not count) the breaker opens: prompts are prefilled on the
  Mac (the log says why once: `the pair is not answering`), and nothing is sent to the pair for `TF_PREFILL_COOLDOWN_S` (default
  30). Then one prompt is the trial: `GET /health` on the GB10 server (2 s) and, over MCDMA, a re-open of the link, then the
  pull. If it succeeds the breaker closes (`remote prefill: the pair answers again`); if not it opens for another
  cool-down. `TF_PREFILL_BREAKER=0` never stops pulling.
* **The connect daemon restarts.** `mcdma-rpcd connect` unlinks and recreates its mailbox at every start, and a mapping
  of the old object never sees the new link. Before each pull the server checks that the mailbox name still maps the
  object it holds (POSIX shared memory on a Mac has no inode, so it writes a nonce into the payload bytes of its own
  mapping, which nothing reads between calls, and looks for it through a new mapping of the name) and, if the daemon
  replaced it, maps the new one (backoff 1 s to 30 s while that is refused) and makes a new client: no server restart
  (`remote prefill: the connect daemon's mailbox was replaced; the link was re-opened`). The producer's listen daemon
  restarting is healed on the GB10 side (the producer attaches again, at most once a second).
* **A link re-links while the producer prefills.** The links are checked before the trigger, and a long prompt's prefill takes
  minutes (the daemons' liveness window is 6 s: a lost control datagram or two drops the link, and it comes back with a new
  generation, which a client made earlier no longer matches). Once the trigger returns, the pull makes every link's client
  afresh and waits up to 30 s for a link that is down at that moment, so the producer's finished export is pulled instead of
  failing with `LinkDown` and the prompt being prefilled again on the Mac (`remote prefill: a link was re-made while the GB10
  prefilled; the pull uses the new one`). A link still down after 30 s fails the pull as before.

### A request that goes away

The server tells a pull when its request ended (the client hung up, or the request was cancelled): a pull waiting at the
gate, for the trigger's reply, or between the decoder's calls stops at once and the export it opened is released, and the
Mac does not log a failure. The GB10's prompt pass itself is **not** stopped. The two ranks run it in lock step (the
follower is driven by the leader's every pass), so abandoning it on one rank would leave the other waiting on a
collective: it runs to its end, the export it makes is dropped by the producer when its time to live (120 s) ends, and a
hand-off request cancelled before its cut is not captured at all.

## 8. Verify

A reply with a long prompt carries, in its `tensorfold` block, `"exact_mode": "remote_prefix"`, `remote_prefix` and
`pulled_prefix` (tokens), plus `remote_route` and the pull's phases (`remote_queue_seconds`, `remote_trigger_seconds`, `remote_pull_seconds`,
`remote_land_seconds`, `remote_tail_seconds`, `remote_bytes`, `remote_tokens_per_second`; see `docs/api.md`), and the server logs one
`remote prefill: N tokens pulled: gate .., trigger .., transfer ..` line for each. A failed pull logs `remote prefill failed (REASON); the prompt is prefilled here` and the reply has
none of those fields. To prove the bytes used the cable, not the LAN, read the producer port's counters around a request:

```
sudo ethtool -S PEER_IF | grep tx_bytes_phy      # grows by about 3.7 KB per prompt token
```

## 9. Running it as services

Recipes for an unattended pair and Mac are in [`deploy/dsv41-disaggregated/`](../deploy/dsv41-disaggregated/): systemd
units for the pair machines, launchd jobs for the Mac, a health check, and `render.sh`, which fills their `@NAME@`
placeholders from one site file per machine.

| file | runs on | what |
|---|---|---|
| `systemd/mcdma-rpcd-listen@.service` | producer (rank 0's host) | the listen daemon, instance name = link name, parameters in `/etc/tensorfold/mcdma-listen-NAME.env` |
| `systemd/tf-pd-link-up.service`, `bin/tf-pd-link-up` | each pair machine | oneshot: the Mac-facing port unmanaged, MTU 9000, 100G, the Mac's static neighbour (section 3) |
| `systemd/tf-pd-prefill-rank0.service`, `-rank1.service` | rank 0's host, the other machine | the pair; `prefill.env` is the tuned profile (section 6), identical on both |
| `macos/dev.tensorfold.pd-restore-links.plist` | Mac | root LaunchDaemon: `restore-rdma.py` for each cabled interface (section 4) |
| `macos/dev.tensorfold.pd-connect.plist` | Mac | the connect daemon |
| `macos/dev.tensorfold.pd-decoder.plist`, `macos/bin/tf-pd-decoder-run` | Mac | the decoder and its supervisor |
| `bin/tf-pd-health` | both sides | the health check |
| `site.conf.example`, `render.sh` | workstation | placeholders and their substitution |

Nothing here is enabled by being installed: the producer's units have no `WantedBy` on the daemon, and the Mac's jobs
only start once bootstrapped (below).

### Render

Copy `site.conf.example` once per machine (the checkpoint path, the Mac's interface and the addresses differ), fill it in,
then `./render.sh head|worker|mac SITE_CONF OUT_DIR`. A placeholder left unset aborts the render, so an unfilled file never
reaches a machine.

### The pair machines (systemd)

```
# on each pair machine, from the rendered tree
sudo install -d /etc/tensorfold /usr/local/libexec/tensorfold
sudo install -m 644 out/systemd/*.service /etc/systemd/system/
sudo install -m 644 out/etc/* /etc/tensorfold/
sudo install -m 755 out/libexec/* /usr/local/libexec/tensorfold/
# the state directory the units name must exist and hold the API key file and the dense-prefill table
systemd-analyze verify /etc/systemd/system/tf-pd-*.service      # and mcdma-rpcd-listen@NAME.service on rank 0's host
sudo systemctl daemon-reload
sudo systemctl enable --now tf-pd-link-up.service               # harmless beside production; at boot, if wanted
```

Start the pair with `systemctl start tf-pd-prefill-rank1` on the follower, then `tf-pd-prefill-rank0` on rank 0's host
(`enable` them for boot only if the pair, not production, should be the default). What the units encode:

* **Order and dependencies.** Rank 0 has `Requires=` and `After=` on `mcdma-rpcd-listen@NAME.service`, which has them on
  `tf-pd-link-up.service`: starting rank 0 starts the daemon first, as the same user, and a daemon that cannot start fails
  rank 0's start instead of leaving it to exit with `no mcdma-rpcd listen daemon serves that link`. Rank 1 needs no daemon.
  Systemd cannot order across machines: start rank 1 first (a rank 0 that starts first waits for it up to
  `TF_TP_PLAN_TIMEOUT`).
* **The NCCL pin is in each rank unit**, as `Environment="NCCL_IB_HCA==PAIR_DEV_A,PAIR_DEV_B"`, not in `prefill.env`: an
  `EnvironmentFile=` overrides `Environment=`, and a profile edit must not be able to drop it.
* **Production stays the default.** Both rank units say `Conflicts=` on the production unit, and the pair holds the same GPUs:
  starting one stops the other, so `systemctl start production-unit` (after stopping the pair) is the way back. Enable only
  one of the two for boot. Neither the oneshot nor the listen daemon conflicts with anything.
* **Restart policy.** Ranks: `Restart=always`, `RestartSec=60`, at most 20 starts in 30 minutes: a slow failure is
  retried, a configuration that fails in seconds stops the unit. A lone rank waits for its peer and systemd cannot tie the two
  machines together, so the follower is restarted by its own policy only. The listen daemon: `Restart=always`, 5 s.
* **The listen daemon is never SIGKILLed.** Its guide forbids it (SIGKILL skips the teardown of its queue pairs), so
  `SendSIGKILL=no`; SIGTERM is an orderly, bounded stop. Rank 0 attaches again by itself when the daemon comes back (it logs
  `attaching again`), so to bounce the daemon use `systemctl kill mcdma-rpcd-listen@NAME`, which restarts only the daemon.
  `systemctl restart` on it restarts rank 0 too (`Requires=`), an eight-minute start.
* **No `PrivateTmp`** on the daemon: its lock file and its status socket are in `/tmp`, and rank 0 attaches through that socket.
* **Logs** go to the journal: `journalctl -u tf-pd-prefill-rank0`; the ready lines are `serving DeepSeek` and `mcdma handoff:
  attached to the listen daemon on link NAME`, and the listen daemon logs `peer ready over roce`.

### The Mac (launchd)

The Mac's jobs are **LaunchDaemons that drop to the serving user** (`UserName`), not LaunchAgents. That reverses a
common assumption, so it is worth checking against Apple's own text. Apple's [TN3179, "Understanding local network
privacy"](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy), section "macOS
considerations", says macOS automatically allows local network access for "any daemon started by `launchd`", "any program
running as root" and "command-line tools run from Terminal or over SSH, including any child processes they spawn", and
then: "The exception for `launchd` daemons doesn't apply to `launchd` agents." An agent is gated by the user's Local
Network grant (per user account, asked once in the GUI), and a bare executable that is not an app bundle has no good way to
hold one; a daemon needs none and also starts at boot with nobody logged in, which an agent (a job of the `gui/UID`
domain) does not. The same note explains the `HostUnreachable` of a detached process: one that `nohup` or `&` reparented
away from its ssh session is neither a command-line tool under ssh nor a daemon started by launchd, so it loses the
exemption. A launchd job is the second thing, whatever its user.

Both LAN-facing programs need it: the connect daemon sends its setup datagrams to the producer's LAN address, and the decoder
opens an HTTP connection to it. `restore-rdma.py` does no LAN traffic, so its job is a plain root LaunchDaemon.

```
sudo install -d -o root -g wheel -m 755 /usr/local/libexec/tensorfold
sudo install -o root -g wheel -m 755 out/libexec/* /usr/local/libexec/tensorfold/
sudo install -d -o SERVICE_USER -m 755 /var/log/tensorfold
sudo install -o root -g wheel -m 644 out/launchd/*.plist /Library/LaunchDaemons/      # launchd refuses a plist not owned by root or writable by others
plutil -lint /Library/LaunchDaemons/dev.tensorfold.pd-*.plist
for job in restore-links connect decoder; do sudo launchctl bootstrap system /Library/LaunchDaemons/dev.tensorfold.pd-$job.plist; done
sudo launchctl print system/dev.tensorfold.pd-connect | grep -E 'state|pid|last exit'
```

A plist in `/Library/LaunchDaemons` is loaded and, with `RunAtLoad`, started at the next boot; `sudo launchctl disable
system/LABEL` keeps one from doing so, and `enable` before `bootstrap` undoes it. To stop a job use `sudo launchctl bootout
system/LABEL`: SIGTERM to the job's process, then SIGKILL to what is left of its process group after `ExitTimeOut`.

* **Order.** There is none in launchd; each job tolerates the others' absence. `restore-links` retries each interface for
  three minutes (the enclosure can enumerate late); the connect daemon exits and is restarted until its CX5 device exists;
  the decoder's supervisor waits up to five minutes for the connect daemon before it starts the server.
* **Nothing is orphaned.** The jobs' programs are the daemon and the supervisor themselves, with no shell in between, so
  launchd is the parent and the process-group leader. `AbandonProcessGroup` is `false` (launchd's default, written out): when
  a job ends, launchd kills what is left of its group. The connect daemon ignores SIGHUP on purpose, which is why the earlier
  runs left it behind when their ssh session died; started by a launchd job it has no session to lose.
  Check a hand-started one is gone first: `pgrep -x mcdma-rpcd`. A daemon holds `/tmp/mcdma-rpc.NAME.lock`, and a second one
  refuses to start while it lives, so a stray daemon makes the job restart in a loop.
* **The server follows the daemon.** The connect daemon creates its mailbox anew at every start, and the server maps it once
  at start-up: after a daemon restart the server would keep the old one and prefill every prompt itself. `tf-pd-decoder-run`
  therefore stops the server (exit 75) when the daemon's pid changes or vanishes, and `KeepAlive` starts it again once the
  new daemon answers.
* **Restart policy.** `KeepAlive` with `SuccessfulExit=false`: any failure restarts after `ThrottleInterval` (15 s daemon,
  30 s decoder); a clean exit stays down, as does `bootout`. `ExitTimeOut` (60 s, 90 s) is how long launchd waits after
  SIGTERM before it sends SIGKILL, so it must outlast the connect daemon's bounded teardown and the server's drain.
* **Logs** are in `LOG_DIR`: `connect.log`, `restore-links.log`, and `decoder.log` (the server's output, appended by the
  supervisor and rotated at each start past 64 MiB, four backups; launchd's own messages for it are in `decoder-launchd.log`).
  launchd holds the connect daemon's file open, so it is not rotated; it logs a line only on a link change.

If a daemon is nonetheless refused LAN access on a Mac, TN3179 describes a system-wide exemption by network, set with `sudo`
and applied after a restart (it loosens the privacy gate for those networks for every program):
`sudo defaults write com.apple.network.local-network AllowedEthernetLocalNetworkAddresses -array "LAN_CIDR" "fe80::/10"`
(`AllowedWiFiLocalNetworkAddresses` for a Wi-Fi LAN).

### Health check

```
tf-pd-health producer [--link NAME]     # rank 0's host: listen daemon up, rank 0 attached, rank 0 /health
tf-pd-health decoder  [--link NAME]     # the Mac: connect daemon up, link up, decoder /health
```

It reads only the daemon's status socket and the server's open `/health` route, and exits 0 (healthy), 1 (degraded: the link
is down or nothing is attached, so the Mac prefills by itself, slower but correct) or 2 (a daemon or a server does not answer).
Run it from a monitor or a timer of your own; no unit here restarts anything on a failed check.

### What this recipe does not do

* Systemd and launchd cannot tie the two pair machines together; a rank 1 whose peer died waits until its own restart policy
  replaces it.
* The pair and the production unit share the GPUs: the switch is an outage, by design.
* Not run on hardware in this form: install it on a spare pair first, and confirm with `tf-pd-health` and one request that
  carries `remote_prefix` (section 8) that the Mac's daemons reach the producer from a boot with nobody logged in.

## 10. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| port has no carrier; `mstlink` opcode 1036 | the cable's EEPROM names no QSFP28 (generic cables) | a cable the card recognises |
| `CQ status=12` in the byte check | cables crossed at the enclosure, or no static neighbour | pair the other Mac interface; install both neighbours |
| `connect failed with status 6` | the exchange was given the cable's interface | give the LAN interface and the pinned peer address |
| the TP pair hangs at start at 100 % CPU, NCCL `RETRY_EXC_ERR` | NCCL autodetected the Mac-facing port | `NCCL_IB_HCA==PAIR_DEV_A,PAIR_DEV_B` |
| rank 0 exits: `no mcdma-rpcd listen daemon serves that link` | the daemon was not running when rank 0 reached the exporter | start the listen daemon first, as the same user |
| `HostUnreachable` from the Mac's server or daemon | a process reparented away from its ssh session or Terminal (`nohup`, `&`) loses macOS's local-network exemption | run both from a launchd job (section 9) or in a live session |
| `link down (the connect end is silent)` in the listen log | the Mac daemon restarted or slept; it reconnects | none, unless it stays down |
| reply has no `remote_prefix` | prompt under `TF_PREFILL_MIN`, producer down, checkpoint hash differs | read the Mac's `remote prefill failed (...)` line |
| `pkill -f PATTERN` ends your ssh command | the pattern matches the command line carrying it | kill by pid |
| the first pass after a hand-off takes 1 to 2 s longer than the tail's size explains | the idle model was un-wired while the pair prefilled | keep `--keep-warm` at its default; not `0` |

## 11. Measured

Two GX10 (GB10, TP=2) and an M5 Ultra Mac Studio (256 GB), the 2.9 bpw EXL3 pack, 100G cables, MCDMA 0.1.18, macOS 27.0:

| item | value |
|---|---|
| link | 100 Gb/s; MCDMA four-way byte checks pass in all three modes |
| round trip, 4 KiB | 17 µs median (`fabric-check`, host memory, one queue pair) |
| streaming, one queue pair, one direction | 10.8 Gbit/s; 28 MiB mailbox reply 17.3 Gbit/s (the Thunderbolt-5 PCIe tunnel is the ceiling: MCDMA documents about 29 Gbit/s writes, 50 Gbit/s reads) |
| hand-off size | 31 MiB for 8,704 tokens (about 3.7 KB a token); 3 requests, all `remote_prefix`, correct answers |
| bytes on the cable | 34.2 MB sent by the producer's port in one request; the LAN carried about 140 KB |

### End to end: time to first token, hand-off against the Mac alone

Same hardware, 2.9 bpw pack, the tuned prefill profile, three prompts per size (distinct text), seconds, median of three
(`max_tokens` 1), Mac alone on the same binary and prompts. The Mac's tail is the part of the prompt past the last
512-token cache boundary of the prefix (96 to 461 tokens). Measured twice: before and after two fixes on the Mac
(the idle model is kept wired while a hand-off is awaited, and the pass is planned from the landed prefix instead of from token 0).

| prompt tokens | prefix handed over | hand-off, before fixes | hand-off, after fixes | Mac alone | after / Mac alone |
|---|---|---|---|---|---|
| 1k | none (below `TF_PREFILL_MIN`) | 1.62 | | 1.62 | 1.00 |
| 4k | 4,096 | 7.11 | 4.93 | 4.61 | 1.07 |
| 8k | 7,680 to 8,192 | 11.89 | 8.88 | 8.34 | 1.06 |
| 16k | 15,872 to 16,384 | 17.90 | 16.10 | 15.78 | 1.02 |
| 30k | 29,184 to 30,208 | 32.17 | 29.12 | 28.96 | 1.01 |

**The hand-off does not shorten time to first token on this hardware.** After the fixes it is level with the Mac alone (within 1 to 7 %):
the pair prefills and exports at about 1.0 to 1.1k tokens a second, which is what the Mac does alone (0.9 to 1.07k), and the pair's
time is 93 to 95 % of the hand-off's. The Mac's landing plus tail is 0.6 to 1.1 s (it follows the tail's length; it was 1.7 to 4.0 s before
the fixes), and the pull adds 0.1 to 0.55 s. A hand-off wins only when the pair prefills substantially faster than the Mac:
the serving profile reaches 2.2 to 2.5k tokens a second but cannot be exported (see above).

What is checked and works: the payload crosses the cable (the producer port's `tx_bytes_phy` grew 3.7 to 3.9 KB per exported
token while the LAN carried about 68 KB); the Mac falls back to its own prefill when the link is down
(`remote prefill: the link is not usable (LinkDown); prompts are prefilled here`, `exact_mode: exact`); a quiet server (12 s and
30 s idle) shows no wake-up penalty; a second turn of a conversation asks the producer again for the same prefix (the producer
answers from its pool in 0.1 s) and the Mac answers in 1.0 s of TTFT. Greedy decoding after a hand-off differs from a Mac-only
prefill within 5 to 10 tokens on three 9 to 10k-token prompts, with fluent, on-topic text on both: that is the `remote_prefix`
contract, not a defect, and no quality benchmark has been run on it. Not measured: a producer HTTP failure mid-transfer,
concurrent requests, quality, `--mcdma-link http` on the final build.

The remaining case for it is capacity, not latency: a pair that is otherwise idle can take long prompts off the Mac's GPU (expected, not measured).
