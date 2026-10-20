# Read-only admission/lock scope review

Measured source:729833615595df66c9f9a438ce9a85061843f010. The frozen serial execution payload47605457…/run3334e2a2… is unchanged by these notes or post-run tools.

The repository's `tools/zig/xpu_guard.sh` uses `/tmp/b70.lock` and scans xe fdinfo VRAM with default8192MB threshold. Related xpu scripts honor `TF_GPU_LOCK`. Its device-memory scan recognizes xe only; it must not be treated as an AMD guard or used to claim applicable AMD GPU-lock qualification.

The native HIP memory admission path is:

- `zig/src/hip/context.zig:47`: `memInfo` calls `hipMemGetInfo`, rejecting invalid totals/free counts.
- `zig/src/hip/admission.zig:50`: `budget` uses device/host availability, reserve and cap rules from shared `cuda_memory` helpers. Environment keys are `TENSORFOLD_MEMORY_RESERVE_GIB` and `TENSORFOLD_HIP_MEMORY_LIMIT_GB`.
- `zig/src/families/nemotron/hip/native.zig:94`: `cardRoom` calls that budget; `check` at104 admits the rank/world/context and refuses an unfitting plan before loading.

These files and xpu_guard.sh are byte-identical between baseline8649 and measured729; exact blob IDs are in `amd_guard_source_review.json`. This is memory admission, not an exclusive GPU lock or a detector for all other model processes. The serial recipe introduces no admission/limit override.

Parent observed both memory environment keys unset in the current SSH launch environment and will capture those exact two keys from actual coordinator/rank environments at serial launch. This observation is not substituted for future process captures. No environment override or unset action is proposed.

The existing shell-script lock scan found the CPU build lock. Applicable existing AMD GPU-lock qualification remains pending. A named-known-process preflight can miss arbitrary Python/renamed GPU programs; a PermissionError is fatal, but that does not make the pattern matcher a detector for every unknown GPU process. Scheduling remains a coordinator assertion plus observed preflights, not a physical-host lease. GPU6–9 are never used or managed by this work.
