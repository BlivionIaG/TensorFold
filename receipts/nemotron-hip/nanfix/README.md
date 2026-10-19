# NaN propagation fix receipts

The corrected stack ends at `8649e9858b38e9ef3dcd8d579879c6512a363050`. The baseline is `55b9084440a40a81ae47ee11d89c06f1ecb3a548`.

`proofs/` retains exact completed proof excerpts, raw-log hashes, measured source identities, source-only comparisons, and the deliberately failing epilogue mutation. The known nonfinite finding belongs only to the shipped plain `affine_dot2_stream`; neither fused relu² kernel is marked known.

`ab/` contains the alternating baseline/fixed/fixed/baseline decode and cold-prefill measurements, exact inputs, request parameters and benchmark clients. Its summary reports measured throughput and equality results.

Historical family, concurrency, memory and checkpoint receipts remain in the parent directories with their original source and binary identities. Source bridges do not relabel those measurements as runs of a new binary.
