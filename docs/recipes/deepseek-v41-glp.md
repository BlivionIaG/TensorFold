# DeepSeek-V4.1 EXL3 control vectors

This fork adds an explicit inference-time steering mechanism. Checkpoint weights
remain unchanged. A supplied vector changes activations and can change generated
tokens; steered output is not the stock model's output. Nothing is enabled without
`--glp`, and no trained vector is distributed with this contribution.

## Attribution and implementation provenance

These concepts and formats are not inventions of this contribution:

- **Matt Suiche (@msuiche), [weightless](https://weightle.ss/):** the weightless
  packaging/runtime concept and **GLP (GGUF Layer Projection)** format. This
  implementation follows the [GLP specification at
  `b181a07d3980fc48abe39d95ddb044459dd0b868`](https://github.com/msuiche/weightless/blob/b181a07d3980fc48abe39d95ddb044459dd0b868/docs/spec/GLP.md),
  including its DeepSeek-specific paired fold/writer contract.
- **Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery,
  Wes Gurnee and Neel Nanda:** the activation-space directional-ablation research,
  [*Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS
  2024](https://arxiv.org/abs/2406.11717). Citing the mechanism does not transfer
  their experiments or efficacy claims to this implementation or checkpoint.
- **llama.cpp's authors and contributors:** the underlying GGUF additive
  control-vector convention, including F32 `direction.N` tensors and their layer
  numbering. GLP extends that convention; additive and projective application are
  different operations, not aliases.
- **TensorFold [#1835](https://github.com/ashhart/TensorFold/pull/1835), source
  `c5947a8d2e9543e6c6a11bd30dc4c43c32f3b0fb`:** the existing TensorFold-native
  Zig container reader and startup approach reused here. Its Qwen additive-only
  implementation and measurements do not qualify this DeepSeek implementation.

The new backend implementation is written for TensorFold from the mathematical
contract and format specification. External serving plugins, engine kernels and
model weights are not vendored or translated here. TensorFold code remains under
its existing Apache-2.0 licence; separately supplied vectors retain their own
licence. This fork's projective mode is not a request to reverse the upstream
maintainer's additive-only decision for #1835.

## Explicit startup

```sh
tensorfold-native serve CHECKPOINT --glp vector.gguf --glp-alpha 1
```

`--glp-alpha` overrides the file's strength. The effective F32 value must be
finite and nonnegative. Zero bypasses the edit, without adding a rounding step.
A missing `--glp` retains the existing unsteered path. Startup reports the exact
file SHA-256 and effective strength, explicitly distinguishing active steering
from a zero-strength, inactive load. Filenames and metadata cannot forge log lines.

The mechanism is immutable for the lifetime of a served model. This is not a
per-request steering switch or a live vector-reload API.

The native EXL3 Metal and CUDA paths are the implementation targets. The separate
experimental FP8 Metal package refuses GLP; its synthetic model is not evidence
for this checkpoint. Qualification requires source-bound runtime receipts, not
just an implementation or a successful build.

Enabled CUDA steering refuses `TF_DSV41_PREFILL=replay` with `GlpUnsupported`:
that path enters decoder layers from a stash, past the fold. Use whole prefill;
an existing replay-performance profile is not an enabled-GLP profile. Keep normal
memory-admission guards: lowering safety floors is not a qualification method.

Existing native Metal admission declares `draft_streams = false`: concurrent
requests share plain target windows. Record actual proposal counters rather than
treating a `draft: true` request as proof of speculative execution. This feature
does not add concurrent MTP support.

## The DeepSeek site is not a Qwen residual hook

DeepSeek-V4.1 carries four BF16 residual copies per token. A direction is H-wide,
not 4H-wide, and `direction.N` names **backbone block N**, zero-based. Layer zero
is not representable by this convention. Drafter/MTP blocks are not steered.

This path requires GLP version 3 and `glp.hook_point = hc_fold_pre_engram`:

1. Fold block N's FFN write into the four-copy stream. In a lagged implementation
   this fold runs at the entrance to block N+1; the final backbone fold precedes
   the head. Apply the edit before Engram, not after the next layer's coefficients
   have already been computed from an unsteered fold.
2. Apply the same direction, operation and strength to block N's attention output
   writer after its complete projection and tensor-parallel reduction, before its
   hyper-connection post mixer.

Both halves are required. A writer-only or fold-only edit is not this GLP site.
The implementation must not relabel `residual_stream_post_layer` or a pre-residual
writer hook as a compatible DeepSeek file.

For copies `x_c`, the canonical stream is `h = mean_c(x_c)`. Rank-one operations
are defined separately:

- `add`: every copy receives `alpha * d`.
- `project`: every copy loses `alpha * dot(h, d) * d`, where `d` is the file's
  unit direction. The attention writer uses the corresponding single-stream edit.

Vectors and reductions are F32; activation storage remains BF16 with round to
nearest, ties to even. Directions are uploaded once and owned until queued work
and captured graphs finish. Steering must not add per-token allocations, host
readbacks or a CPU arithmetic fallback.

## Fail closed

Malformed mode/version/rank/scale metadata must not silently become another
operation. This rank-one path refuses higher-rank bases, unsupported scaling
metadata, wrong hooks, duplicate/out-of-range layer IDs, incompatible widths,
nonfinite direction values and a mismatched claimed tensor digest. Projective
unit-vector validation must not silently change the bytes being applied.

Checkpoint provenance is a separate gate from matching H and layer count. A local
conversion may not retain an upstream repository/revision; that absence is not
proof that a published vector was derived from these weights. Do not restamp a
foreign vector merely to pass a loader check. Matching geometry establishes
loadability, not behavioural validity after EXL3 quantization.

## State and a split producer/consumer

A remote prefix is valid only for the same checkpoint and steering policy. The
existing handoff model identity is extended with the complete vector file hash,
mode/site and effective F32 strength. The old identifier stays unchanged when
steering is absent. Missing or different steering must fail manifest validation
before any prefix state is imported, including file-based handoffs. Tensor-parallel
ranks must also agree before execution; configuring only rank zero is not support.

A file loaded at zero strength remains a distinct policy namespace, even though
its output is a no-op. It is not a state-identity alias for an absent file.
Positive and negative F32 zero use the same effective-strength identity.

A successful mixed run does not establish equality with native Metal prefill.
The existing CUDA-to-Metal numerical/state exactness limitations remain separate;
this contribution does not waive them or revive the declined upstream consumer.

## Qualification and later vector publication

Keep two claims separate:

- **Mechanism correctness:** independent arithmetic references and planted
  controls; actual prefill/decode execution; default-off/zero identity; drafted
  versus plain, concurrent versus solo, and restored versus fresh state under the
  same policy; matching/mismatched producer-consumer identities. Record exact
  source/binary/vector hashes and before/after performance on each reached device.
- **Vector efficacy:** checkpoint- and quantization-specific derivation and
  calibration on held-out prompts. A synthetic direction proving arithmetic or
  changing an output proves neither a useful steer nor preserved capability.

Trained vectors will be published separately. Before that release, include the
original concept credits above, vector author and licence, exact base revision
and EXL3 conversion identity, tensor/file digests, layer IDs and hook, derivation
method and prompt provenance, measured alpha ladder, norm-matched/random-direction
controls, held-out capability results and the limitations of those measurements.
Do not publish checkpoint tensors, private prompts or claims borrowed from another
project's benchmarks.

Device and model receipts belong to the source-bound draft PR. An unrun platform,
precision setting or workload is not qualified by a successful run elsewhere.

## Numerical contract and stress fixtures

For FP32 unit roundoff `u = 2^-24`, use `gamma(n) = n*u/(1-n*u)`.
The projection error scale uses `A_j = mean_c(abs(x_cj))` and
`D = sum_j(abs(d_j)*A_j)`, not the absolute value of the copies' mean:
opposite-signed copies can cancel without eliminating mean-rounding error.
Scale each BF16 copy by one quarter before adding, rather than overflowing a
four-copy sum and scaling it afterwards.

- **Metal:** the dot's conservative allowance is
  `gamma(ceil(H/256)+16)*D`. Including strength, correction and subtraction gives
  registered depth `ceil(H/256)+20` against
  `abs(x_ci)+alpha*abs(d_i)*D`, followed by the BF16 output allowance.
  Add mode accounts for its FP32 product/add and BF16 output separately.
- **CUDA:** the dot uses conservative `gamma(50)*D`; the served width uses only
  sixteen per-thread terms, below that static budget. Three post-dot roundings
  and the BF16 output half-ulp are separate. Writer bounds include full-writer
  BF16 rounding; final bounds include collapse tie flips and the RMS chain
  (depth 170, five post-sum operations). Stress cases retain their exact-sum
  allowances instead of using a loose random-input gamma allowance.

The CUDA projection stress case has unit terms except at its first and last
columns, where `m=2048` and `d=+1024,-1024` give opposite `2^21` terms.
All FP32 partial sums stay below `2^24`, so every tested FP32 order gives `H-2`.
In each tested BF16 order, the unit groups disappear into those large partials;
FP16 overflows. The resulting miss exceeds the required 100-times exact
allowance. The host witness exercises sequential, pairwise and strided orders.
These unnormalized directions stress the low-level arithmetic, not the file
loader: a served projective file must still pass unit-norm validation.

The CUDA final stress case uses pre-mix `(1,0,0,0)` and integer copies in
`[16,57]`. Its FP32 square sum is exact and below `2^24`, while every tested
FP16 order overflows. A pairwise BF16 RMS sum cannot guarantee a 100-times
output error, so that specific BF16 discrimination is declared unprovable
rather than given a guessed threshold.

Paired-program controls use real nonidentity mHC posts at width 5120. They
establish sensitivity to missing or shifted edits, not a trusted full FFN,
drafter or model-quality reference. Production-emitter placement, device
arithmetic, and actual served-model state checks remain distinct evidence.

Metal launches reserve 256 threads and 32 bytes of static threadgroup storage.
CUDA launches use 320 threads; the final additionally requests `2*H` dynamic
shared bytes. Register and static-shared usage must come from the actual
compiled CUDA image, not estimates or another compiler's receipt.

### Tap rounding and cancellation

The CUDA tap averages the edited BF16 copies, so its FP64 reference includes
that intermediate quantization. Let `q_c=RNE16(y_c)`, `m=mean(q_c)`,
`E=mean(copy_allowance_c+abs(y_c-q_c))` and `A=mean(abs(q_c))`.
The FP32 mean error is bounded by `e=E+gamma(3)*(A+E)`. Its final BF16 tap
allowance is `e+2^-8*(abs(m)+e)+2^-133`.

Every device result must match the independently documented FP32 order and stay
inside the FP64 error bound. ULP distance from a once-rounded FP64 result is
diagnostic, not a blanket one-ULP gate: a rounded FP32 product followed by
cancellation can legitimately produce a larger ULP distance.

## Native prefix-cache qualification

For plain completions without chat history, the current native policy creates
tail and end snapshot marks only when the prompt exceeds 4096 tokens.
An identical 2048-token request therefore exercises no restoration, even when
the cache has enough budget. Use a longer fixture, such as 8192 tokens, and
require nonzero `cached_tokens`; matching replies alone do not prove reuse.
Compare fresh and restored token IDs under the same immutable policy, including
actual solo drafting and seeded sampling, and keep the cache-off oracle separate.
