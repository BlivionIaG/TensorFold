# Current upstream CI classification

All 22 current published heads were checked. All 110 checks completed:87 successful and23 failed. Every Windows failure has exactly the independently established52 baseline-equivalent compiler diagnostics, with zero additional/missing diagnostics. No explicit abnormal test-runtime result appears in the final Windows failure summaries. This does not make the Windows jobs green.

The only non-Windows failure is PR1085's desktop guest output-limit fixture:820/821 test files passed, and the guest returned an operation-not-permitted refusal instead of the expected truncated result. Its actual tested merge has the identical complete app and workflow trees as upstream main d101. Historical PR1092's actual tested merge had those same inputs and the identical failure. This supports a pre-existing main-input/environment-sensitive app failure; the exact failing syscall is not proved. No HIP source fix or test weakening is justified by these logs.

No workflow rerun, GPU/model/benchmark work, source change or Ready action was performed. Raw logs and API metadata remain in the private audit folder; public excerpts normalize the CI user root. Tree objects, source/head/job identities and raw-log hashes support the scoped classification.
