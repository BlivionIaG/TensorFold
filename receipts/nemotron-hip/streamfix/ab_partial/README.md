# Partial A/B: completed drafts-on quartets

Measured production sources: old8649e985 → newcc18314d. Complete replies and token hashes match across old/new/new/old on both rigs. This is a material decode regression; full A/B and remaining exact-head qualification are still pending.

- W7800tp2 fibonacci-raw: 255.14 → 161.62 tok/s (-36.65%).
- W7800tp2 gpu-chat-no-think: 215.29 → 141.85 tok/s (-34.11%).
- V620tp1 fibonacci-raw: 180.17 → 121.46 tok/s (-32.58%).
- V620tp1 gpu-chat-no-think: 156.56 → 107.25 tok/s (-31.50%).

Raw benchmark JSON, all per-hold executable/library/checkpoint identity sidecars and SHA bindings are included. Personal home prefixes only were redacted from identity paths; raw sidecar hashes remain. No throughput threshold or full qualification pass is asserted.
