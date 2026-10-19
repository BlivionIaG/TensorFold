# NaN fix A/B receipts

A/B 55b90844→8649e985, old/new/new/old; V620tp1: on/fibonacci-raw 180.32→180.47 tok/s (+0.08%), on/gpu-chat-no-think 155.95→156.50 tok/s (+0.35%), off/fibonacci-raw 133.46→133.76 tok/s (+0.23%), off/gpu-chat-no-think 133.27→133.51 tok/s (+0.18%), cold8192 first TTFT 3.458→3.460s (+0.03%), cold32768 first TTFT 17.423→17.470s (+0.27%); W7800tp2: on/fibonacci-raw 255.59→256.31 tok/s (+0.28%), on/gpu-chat-no-think 214.07→216.45 tok/s (+1.11%), off/fibonacci-raw 148.86→148.65 tok/s (-0.14%), off/gpu-chat-no-think 148.29→148.00 tok/s (-0.20%), cold8192 first TTFT 1.727→1.725s (-0.16%), cold32768 first TTFT 8.912→8.954s (+0.47%); complete replies and token_sha equal in every quartet.

Full hold-order values, all three cold request timings and medians, source commits, binary hashes and token hashes are in summary.json and manifest.json.
