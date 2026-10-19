# Plain-stream correction A/B receipts

A/B 8649e985→cc18314d, old/new/new/old; W7800tp2: on/fibonacci-raw 255.14→161.62 tok/s (-36.65%), on/gpu-chat-no-think 215.29→141.85 tok/s (-34.11%), off/fibonacci-raw 148.33→72.73 tok/s (-50.97%), off/gpu-chat-no-think 146.65→71.78 tok/s (-51.05%), cold8192 first TTFT 1.733→1.733s (+0.04%), cold32768 first TTFT 9.151→8.973s (-1.94%); V620tp1: on/fibonacci-raw 180.17→121.46 tok/s (-32.58%), on/gpu-chat-no-think 156.56→107.25 tok/s (-31.50%), off/fibonacci-raw 133.16→90.93 tok/s (-31.71%), off/gpu-chat-no-think 132.90→90.81 tok/s (-31.67%), cold8192 first TTFT 3.464→3.462s (-0.05%), cold32768 first TTFT 17.495→17.458s (-0.21%); complete replies and token_sha equal in every quartet.

Full hold-order values, all three cold request timings and medians, source commits, binary hashes and token hashes are in summary.json and manifest.json.

All eight condition quartets and 32 benchmark holds validate complete replies/token hashes; cold requests have zero cached tokens. This is a material decode regression, not readiness or complete-stack qualification. Measured sources remain8649e985→cc18314d; review1596156 uses the separately supplied seven-line fixture-only source bridge. No runtime result is relabelled as review-head measurement.

Cold32k W7800 first TTFT varied across holds: baseline8.900s then9.402s; corrected8.976s and8.970s. The reported mean change (-1.94%) must be read with that within-run drift, not as an established prefill speedup. All raw timings and medians are retained.
