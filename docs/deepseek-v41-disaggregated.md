# DeepSeek-V4.1 disaggregated prefill and decode (GB10 pair + Apple Silicon)

A pair of NVIDIA GB10 machines (two DGX Spark or ASUS Ascent GX10, tensor-parallel over CUDA) **prefill** the prompt; an
Apple Silicon Mac (Metal, EXL3) **decodes** it and serves the API. The prefill state crosses on a ConnectX-5 cable through
[MCDMA](https://github.com/ashhart/MCDMA) (RDMA from macOS), or over plain HTTP where there is no such cable.

This page is the operator recipe. The wire format, the library and the exactness contract are in
[deepseek-v41-handoff.md](deepseek-v41-handoff.md); the model lanes are in [deepseek-v41.md](deepseek-v41.md).

> A hand-off prefix is outside the exactness contract: it is opt-in (`--prefill-from`), labelled `remote_prefix` in the
> reply, and decode after it is not bit-identical to a Mac-only prefill. Everything below keeps the Mac able to prefill
> by itself: a producer that is down, slow or wrong only costs the Mac its normal prefill.

## What runs where

```
   client ──HTTP──▶  Mac (Metal): `tensorfold-native serve ... --prefill-from http://PRODUCER:PORT --mcdma-link NAME`
                       │  1. trigger the export     POST /v1/completions + kv_transfer_params   (HTTP, LAN)
                       │  2. pull the state         MKVH protocol 2 over the link NAME          (RDMA, CX5 cable)
                       │  3. land it, prefill the tail, decode
                       ▼
   GB10 pair (CUDA, TP=2): rank 0 `tensorfold-dsv41 serve ... --mcdma-produce NAME`, rank 1 `tf-dsv41 follow`
                       prefills the prompt to its last-but-one token, exports rank 0's state, holds it until pulled
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

# Mac, in a session that holds the "Local Network" permission (a Terminal or an ssh session; a detached process gets
# HostUnreachable to LAN hosts)
mcdma-rpcd connect NAME,en0/fe80::PRODUCER_LAN_LINK_LOCAL,18620,rdma_mcrdmaN,0,4096,4,64
```

The listen daemon logs `peer ready over roce` once the Mac connects. `NAME` is the link name used below.

## 6. The prefill pair

Rank 1 is `tf-dsv41 follow`; rank 0 is `tensorfold-dsv41 serve` with one more flag. The pair needs a profile whose
prefill state can be exported: replicated KV, whole-prompt prefill, a paged pool. It must **not** run
`TF_DSV41_PREFILL=replay` (decoder layers' state for the prefix never exists), `TF_DSV41_PREFILL_PIECES=1` (a prompt in
pieces never stands at the cut) or `TF_DSV41_KV_SPLIT=1` (half the pages are on the other rank); the server refuses to
start, by name, with any of them.

```
# both machines
export NCCL_IB_HCA==PAIR_DEV_A,PAIR_DEV_B TF_DSV41_FORMS_INPLACE=1 TF_DSV41_SAMPLING=1 TF_DSV41_POOL_TOKENS=POOL
# (PREFILL_PROFILE: the tuned prefill settings are in the next section)
# rank 0 only
tensorfold-dsv41 serve $MODEL --host 0.0.0.0 --port 8000 --name DeepSeek-V4.1-Flash-TF --no-thinking \
    --mcdma-produce NAME --api-key-file $KEYFILE
```

Do **not** set `TF_DSV41_HANDOFF_DUMP` in service: it writes each export to disk on the engine thread.

Rank 0 logs `mcdma handoff: attached to the listen daemon on link NAME` and `exporting DeepSeek-V4.1 states as MKVH protocol 2
on link NAME`; a start refusal names the knob.

## 7. The Mac decoder

```
TF_PREFILL_API_KEY_FILE=$KEYFILE tensorfold-native serve $PACK --engram-stream-dir $ENGRAM \
    --chat-template $PACK/chat_template.jinja --context 32768 --prompt-cache-gib 4 \
    --prefill-from http://PRODUCER:8000 --mcdma-link NAME
```

`--prefill-from` is the producer's HTTP address (the trigger); `--mcdma-link NAME` is the daemon link (the payload).
`TF_DSV41_PREFILL_STEP=4096` is the Mac's own tail-prefill step. A prompt shorter than `TF_PREFILL_MIN` is prefilled on the
Mac.

## 8. Verify

A reply with a long prompt carries, in its `tensorfold` block, `"exact_mode": "remote_prefix"`, `remote_prefix` and
`pulled_prefix` (tokens). A failed pull logs `remote prefill failed (REASON); the prompt is prefilled here` and the reply has
none of those fields. To prove the bytes used the cable, not the LAN, read the producer port's counters around a request:

```
sudo ethtool -S PEER_IF | grep tx_bytes_phy      # grows by about 3.7 KB per prompt token
```

## 9. Running it as services

Start order matters: the producer's listen daemon, then rank 1, then rank 0 (which attaches to the daemon while it
starts), then, at any time, the Mac's connect daemon and its server. Stop in the reverse order.

On each pair machine, a boot unit restores what a reboot loses (the static neighbour and the unmanaged port; the Mac's MAC and
link-local address are fixed for its CX5):

```
# /usr/local/bin/mac-link-up  (run by a oneshot unit After=network-online.target)
nmcli dev set PEER_IF managed no
ip link set dev PEER_IF mtu 9000 up
ip -6 neigh replace MAC_GID lladdr MAC_HWADDR nud permanent dev PEER_IF
```

```
# mcdma-rpcd-listen.service on the producer machine, User=the serving user, LimitMEMLOCK=infinity
ExecStart=/path/to/mcdma-rpcd listen NAME PEER_DEV PEER_GID_INDEX 4096 LAN_IF/fe80::MAC_LAN_LINK_LOCAL:18620 4 64
# the rank-0 unit: Requires=mcdma-rpcd-listen.service After=mcdma-rpcd-listen.service, and the NCCL_IB_HCA pin in its Environment=
```

On the Mac the daemon, then the server, run in the user's login session (a launchd *agent*, not a system daemon, so that
macOS Local Network access applies to the logged-in user). The Mac server falls back to its own prefill whenever the producer
is down, so the units do not depend on each other.

The Mac's restore command (`tools/restore-rdma.py`) and the peer's neighbour are lost on reboot of either side: run both at boot.


## 10. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| port has no carrier; `mstlink` opcode 1036 | the cable's EEPROM names no QSFP28 (generic cables) | a cable the card recognises |
| `CQ status=12` in the byte check | cables crossed at the enclosure, or no static neighbour | pair the other Mac interface; install both neighbours |
| `connect failed with status 6` | the exchange was given the cable's interface | give the LAN interface and the pinned peer address |
| the TP pair hangs at start at 100 % CPU, NCCL `RETRY_EXC_ERR` | NCCL autodetected the Mac-facing port | `NCCL_IB_HCA==PAIR_DEV_A,PAIR_DEV_B` |
| rank 0 exits: `no mcdma-rpcd listen daemon serves that link` | the daemon was not running when rank 0 reached the exporter | start the listen daemon first, as the same user |
| `HostUnreachable` from the Mac's server or daemon | macOS Local Network permission for a detached process | run both in a Terminal or ssh session |
| `link down (the connect end is silent)` in the listen log | the Mac daemon restarted or slept; it reconnects | none, unless it stays down |
| reply has no `remote_prefix` | prompt under `TF_PREFILL_MIN`, producer down, checkpoint hash differs | read the Mac's `remote prefill failed (...)` line |
| `pkill -f PATTERN` ends your ssh command | the pattern matches the command line carrying it | kill by pid |

## 11. Measured

Two GX10 (GB10, TP=2) and an M5 Ultra Mac Studio (256 GB), the 2.9 bpw EXL3 pack, 100G cables, MCDMA 0.1.18, macOS 27.0:

| item | value |
|---|---|
| link | 100 Gb/s; MCDMA four-way byte checks pass in all three modes |
| round trip, 4 KiB | 17 µs median (`fabric-check`, host memory, one queue pair) |
| streaming, one queue pair, one direction | 10.8 Gbit/s; 28 MiB mailbox reply 17.3 Gbit/s (the Thunderbolt-5 PCIe tunnel is the ceiling: MCDMA documents about 29 Gbit/s writes, 50 Gbit/s reads) |
| hand-off size | 31 MiB for 8,704 tokens (about 3.7 KB a token); 3 requests, all `remote_prefix`, correct answers |
| bytes on the cable | 34.2 MB sent by the producer's port in one request; the LAN carried about 140 KB |

PREFILL_RESULTS_PLACEHOLDER
