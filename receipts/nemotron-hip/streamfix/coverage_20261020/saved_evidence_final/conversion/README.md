# Actual derived FP16 checkpoint conversion

The reviewed converter890ab464 executed CPU-only on the exact original MLX checkpoint revisiond9d758fb83953437f7263256b0d96157e2a348b8. It completed exit0 in939.569seconds, Python3.12.14/NumPy2.3.5, isolated/no-site with no Torch/vLLM/Transformers modules loaded. No GPU/model-server or inference ran.

All62 output shards were independently streamed through SHA256 and their headers/index checked:6513 output tensors,65,826,538,624 tensor bytes, no temporary shards. Both config dtype fields are float16 and quantization metadata removed; nonquantizedBF16/F32 payloads remain preserved. The original five header/config/revision/stat bindings stayed unchanged; this is not a whole-original-payload rehash. Per-source read-order stream hashes and all output hashes remain in conversion_manifest.json. Model weights are deliberately absent from this evidence package.

Full affine storage rounding:32,904,044,544 values;229,489 nonzero differences; maxabs0.000244140625; meanabs1.2982013017950345e-10; RMSE5.980788205708283e-08. These compare reconstructed FP32 weights with once-rounded FP16 storage, not model logits or inference accuracy. The earlier bounded independent row audit is separate. MTP tensors are preserved under explicit namespace; stock base loader intentionally excludes them.

Actual owned child process group is gone. No stock weight load, runtime support, device fit, reference inference, resolution of five historical FP32 findings, or full CONTRIBUTING qualification is claimed. Fresh disk/RAM snapshots were admission observations, not reservations. Exact output/native arithmetic equality is not asserted. Original metadata/log/provenance hashes are retained; only personal home-path strings are normalized in public copies.
