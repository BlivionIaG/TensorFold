# Prespecified M4 controlled comparison

Status: protocol and code frozen before collection; no performance result claimed.

PR head `17649539d3cf1375dd711455fd2001a3f75fe725` adds manual dispatch for unchanged host/format checks and documentation. Runtime source/build objects match qualified source `afe8bda62562519b4fd84ee9a956392a21f26a60`. The rebuilt native binary remains byte-identical: `c03ab9600d36c32dd1d80b7d94bf9dae15d7256fa8f41df7f23b488c19b43a2c`. Earlier frozen qualification remains labelled sourceAFE; it is not overwritten.

[Protocol](protocol.json): six rounds, independent plain-before/plain-after controls around BF16/8-bit; every original code/chat,T0/T1,width1/4/8/16 cell;512tokens,context4096,parallel16,cache0. Two fixed complete warmup sweeps precede each measured sweep. No adaptive sample count, slow-sample discard, power/pressure override or retry for a favorable sign.

Acceptance requires A/A equivalence within1% in every cell and every candidate's median paired loss<1% plus the prespecified one-sided bound. The Student-t/independent-round model is an assumption, not a universal error floor. Every attempted launch is retained.

[Collector](run-controlled-study.py), [analyzer](analyze-controlled-study.py), [source/binary/model/client binding](study-binding.json), [runtime identity](runtime-source-identity.json), [CI-only patch](ci-dispatch.patch), [hash manifest](SHA256.json).

Actual branch-targeted CI dispatches: [No GPU](https://github.com/BlivionIaG/TensorFold/actions/runs/41725301465), [Zig format](https://github.com/BlivionIaG/TensorFold/actions/runs/41725301489). Dispatch acceptance is not claimed as a green result. No re-review or merge request.
