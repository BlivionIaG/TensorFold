# Summary: every run

Every hold of every rig, in run order (see `README.md` for binaries, inputs and commands). Times are each request's `ttft_s` as `bench_cache.py` measured it, in seconds; decode rates are `bench_openai.py`'s medians in tok/s. The HIP server logs no heat waits, so these times are gross and net alike. `family` prefill at 2,048 to 65,536 tokens is the #46 holds of `fusions/` (h3, h4); 131,072 and 262,080 are `family/*/f2_pr46_long`.


## fusions


### 1 × V620, tp 1


#### Decode, drafts on: 1 × V620, tp 1

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | h1 #38 | h2 #44 | h3 #46 | h4 #46 | h5 #44 | h6 #38 |
|---|---:|---:|---:|---:|---:|---:|
| fibonacci-raw | 156.53 (156.29, 156.70, 156.53) | 175.48 (175.33, 175.51, 175.48) | 178.37 (178.18, 178.37, 178.40) | 178.37 (178.10, 178.41, 178.37) | 175.69 (175.46, 175.73, 175.69) | 154.67 (153.97, 154.69, 154.67) |
| gpu-chat-no-think | 132.97 (132.99, 132.97, 132.93) | 151.41 (151.41, 151.41, 151.39) | 154.63 (154.70, 154.63, 154.61) | 154.73 (154.73, 154.89, 154.68) | 151.64 (151.64, 151.65, 151.62) | 132.95 (132.97, 132.95, 132.86) |

fibonacci-raw: mean of medians #38 155.60, #44 175.59, #46 178.37; #46 / #38 +14.6 %; #44 / #38 +12.8 %. `token_sha` per tree: #38 5bb240972d6b; #44 5bb240972d6b; #46 5bb240972d6b.

gpu-chat-no-think: mean of medians #38 132.96, #44 151.52, #46 154.68; #46 / #38 +16.3 %; #44 / #38 +14.0 %. `token_sha` per tree: #38 9f806c1e0e42; #44 9f806c1e0e42; #46 9f806c1e0e42.

#### Decode, drafts off: 1 × V620, tp 1

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | h1 #38 | h2 #44 | h3 #46 | h4 #46 | h5 #44 | h6 #38 |
|---|---:|---:|---:|---:|---:|---:|
| fibonacci-raw | 110.51 (110.51, 110.52, 110.49) | 126.02 (126.02, 126.01, 126.02) | 133.03 (133.05, 133.03, 132.98) | 133.00 (133.05, 133.00, 132.86) | 126.07 (126.07, 126.08, 126.07) | 110.28 (110.28, 110.28, 110.30) |
| gpu-chat-no-think | 110.40 (110.40, 110.40, 110.40) | 125.68 (125.82, 125.68, 125.45) | 132.87 (132.87, 132.90, 132.83) | 132.73 (132.71, 132.73, 132.82) | 125.86 (125.86, 125.86, 125.85) | 110.10 (110.10, 110.11, 110.09) |

fibonacci-raw: mean of medians #38 110.40, #44 126.04, #46 133.02; #46 / #38 +20.5 %; #44 / #38 +14.2 %. `token_sha` per tree: #38 5bb240972d6b; #44 5bb240972d6b; #46 5bb240972d6b.

gpu-chat-no-think: mean of medians #38 110.25, #44 125.77, #46 132.80; #46 / #38 +20.5 %; #44 / #38 +14.1 %. `token_sha` per tree: #38 9f806c1e0e42; #44 9f806c1e0e42; #46 9f806c1e0e42.

#### Cold prefill, #38 (base) against #46 (branch): 1 × V620, tp 1

Hold order: h1 #38, h3 #46, h4 #46, h6 #38.

| prompt tokens | #38 median `ttft_s` | #46 median `ttft_s` | branch / base | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|---|---|
| 2,048 | 0.829 (6) | 0.826 (6) | 0.996 | h1: 0.824, 0.829, 0.829; h3: 0.820, 0.826, 0.826; h4: 0.820, 0.826, 0.826; h6: 0.826, 0.832, 0.832 | all 0 (16) | c62de829e274093e | equal |
| 8,192 | 3.550 (6) | 3.535 (6) | 0.996 | h1: 3.540, 3.547, 3.549; h3: 3.527, 3.534, 3.538; h4: 3.528, 3.535, 3.539; h6: 3.550, 3.559, 3.563 | all 0 (16) | 5c418c36f5e1f55d | equal |
| 32,768 | 17.954 (6) | 17.894 (6) | 0.997 | h1: 17.896, 17.941, 17.958; h3: 17.848, 17.884, 17.909; h4: 17.848, 17.903, 17.946; h6: 17.949, 17.991, 18.008 | all 0 (16) | 132876962ac928f3 | equal |
| 65,536 | 45.857 (6) | 45.759 (6) | 0.998 | h1: 45.776, 45.813, 45.845; h3: 45.689, 45.749, 45.770; h4: 45.750, 45.768, 45.770; h6: 45.870, 45.885, 45.883 | all 0 (16) | 60049d7f13388407 | equal |

### 2 × V620, tp 2


#### Decode, drafts on: 2 × V620, tp 2

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | h1 #38 | h2 #44 | h3 #46 | h4 #46 | h5 #44 | h6 #38 |
|---|---:|---:|---:|---:|---:|---:|
| fibonacci-raw | 182.40 (182.40, 180.06, 182.85) | 202.81 (202.81, 200.33, 203.63) | 213.41 (216.52, 212.68, 213.41) | 212.65 (212.65, 209.55, 213.13) | 209.93 (209.93, 205.89, 211.20) | 187.93 (187.93, 184.85, 188.18) |
| gpu-chat-no-think | 148.57 (151.54, 147.07, 148.57) | 172.46 (174.08, 172.46, 172.26) | 177.52 (176.77, 178.13, 177.52) | 177.07 (177.92, 177.07, 176.28) | 167.93 (167.64, 167.93, 167.96) | 148.32 (146.60, 148.32, 150.55) |

fibonacci-raw: mean of medians #38 185.16, #44 206.37, #46 213.03; #46 / #38 +15.0 %; #44 / #38 +11.5 %. `token_sha` per tree: #38 c91869ce1808; #44 c91869ce1808; #46 c91869ce1808.

gpu-chat-no-think: mean of medians #38 148.45, #44 170.20, #46 177.30; #46 / #38 +19.4 %; #44 / #38 +14.7 %. `token_sha` per tree: #38 2fdc4c22dd60; #44 2fdc4c22dd60; #46 2fdc4c22dd60.

#### Decode, drafts off: 2 × V620, tp 2

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | h1 #38 | h2 #44 | h3 #46 | h4 #46 | h5 #44 | h6 #38 |
|---|---:|---:|---:|---:|---:|---:|
| fibonacci-raw | 108.68 (108.68, 108.92, 108.26) | 118.21 (116.78, 118.69, 118.21) | 120.58 (120.07, 121.56, 120.58) | 121.95 (122.60, 121.95, 121.21) | 115.10 (115.55, 115.10, 114.79) | 106.79 (105.83, 107.26, 106.79) |
| gpu-chat-no-think | 107.02 (107.02, 106.78, 108.93) | 118.82 (115.97, 118.82, 119.01) | 120.99 (119.94, 120.99, 121.09) | 120.91 (120.91, 119.86, 121.42) | 115.07 (115.00, 115.07, 115.79) | 106.45 (106.42, 106.45, 107.61) |

fibonacci-raw: mean of medians #38 107.73, #44 116.65, #46 121.26; #46 / #38 +12.6 %; #44 / #38 +8.3 %. `token_sha` per tree: #38 c91869ce1808; #44 c91869ce1808; #46 c91869ce1808.

gpu-chat-no-think: mean of medians #38 106.74, #44 116.94, #46 120.95; #46 / #38 +13.3 %; #44 / #38 +9.6 %. `token_sha` per tree: #38 2fdc4c22dd60; #44 2fdc4c22dd60; #46 2fdc4c22dd60.

#### Cold prefill, #38 (base) against #46 (branch): 2 × V620, tp 2

Hold order: h1 #38, h3 #46, h4 #46, h6 #38.

| prompt tokens | #38 median `ttft_s` | #46 median `ttft_s` | branch / base | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|---|---|
| 2,048 | 0.564 (6) | 0.568 (6) | 1.006 | h1: 0.556, 0.553, 0.556; h3: 0.564, 0.565, 0.566; h4: 0.569, 0.571, 0.572; h6: 0.573, 0.572, 0.575 | all 0 (16) | c62de829e274093e | equal |
| 8,192 | 2.394 (6) | 2.413 (6) | 1.008 | h1: 2.352, 2.350, 2.355; h3: 2.397, 2.398, 2.406; h4: 2.419, 2.423, 2.429; h6: 2.433, 2.436, 2.442 | all 0 (16) | 51c60772acc9542f | equal |
| 32,768 | 11.857 (6) | 11.977 (6) | 1.010 | h1: 11.578, 11.650, 11.709; h3: 11.837, 11.923, 12.015; h4: 11.954, 12.000, 12.005; h6: 12.005, 12.054, 12.095 | all 0 (16) | 132876962ac928f3 | equal |
| 65,536 | 29.845 (6) | 29.764 (6) | 0.997 | h1: 29.394, 29.652, 29.786; h3: 29.693, 29.755, 29.752; h4: 29.882, 29.774, 29.827; h6: 29.941, 29.904, 29.940 | all 0 (16) | 79ee88611a8817b1 | equal |

### 2 × W7800, tp 2


#### Decode, drafts on: 2 × W7800, tp 2

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | h1 #38 | h2 #44 | h3 #46 | h4 #46 | h5 #44 | h6 #38 |
|---|---:|---:|---:|---:|---:|---:|
| fibonacci-raw | 211.63 (206.03, 211.63, 214.26) | 246.04 (246.40, 245.91, 246.04) | 244.68 (237.16, 246.19, 244.68) | 244.11 (244.11, 243.81, 244.35) | 239.99 (236.82, 241.59, 239.99) | 201.72 (208.44, 201.72, 198.10) |
| gpu-chat-no-think | 175.61 (175.81, 175.61, 175.41) | 208.64 (208.06, 209.68, 208.64) | 204.18 (204.40, 204.18, 203.82) | 206.94 (195.93, 207.38, 206.94) | 203.24 (199.04, 203.48, 203.24) | 172.13 (172.27, 168.04, 172.13) |

fibonacci-raw: mean of medians #38 206.67, #44 243.02, #46 244.39; #46 / #38 +18.3 %; #44 / #38 +17.6 %. `token_sha` per tree: #38 0808cf694e6f; #44 0808cf694e6f; #46 0808cf694e6f.

gpu-chat-no-think: mean of medians #38 173.87, #44 205.94, #46 205.56; #46 / #38 +18.2 %; #44 / #38 +18.4 %. `token_sha` per tree: #38 25c162db90ad; #44 25c162db90ad; #46 25c162db90ad.

#### Decode, drafts off: 2 × W7800, tp 2

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | h1 #38 | h2 #44 | h3 #46 | h4 #46 | h5 #44 | h6 #38 |
|---|---:|---:|---:|---:|---:|---:|
| fibonacci-raw | 132.21 (132.21, 132.42, 132.14) | 143.92 (144.20, 143.92, 143.65) | 145.98 (146.01, 145.98, 145.64) | 145.87 (145.87, 145.69, 146.28) | 142.09 (142.22, 142.09, 141.83) | 126.91 (127.03, 126.87, 126.91) |
| gpu-chat-no-think | 132.15 (132.15, 132.34, 132.14) | 143.25 (143.93, 143.24, 143.25) | 144.98 (145.24, 144.73, 144.98) | 145.67 (145.99, 145.67, 145.55) | 140.58 (141.09, 140.58, 140.35) | 126.49 (126.78, 126.39, 126.49) |

fibonacci-raw: mean of medians #38 129.56, #44 143.00, #46 145.92; #46 / #38 +12.6 %; #44 / #38 +10.4 %. `token_sha` per tree: #38 0808cf694e6f; #44 0808cf694e6f; #46 0808cf694e6f.

gpu-chat-no-think: mean of medians #38 129.32, #44 141.92, #46 145.32; #46 / #38 +12.4 %; #44 / #38 +9.7 %. `token_sha` per tree: #38 25c162db90ad; #44 25c162db90ad; #46 25c162db90ad.

#### Cold prefill, #38 (base) against #46 (branch): 2 × W7800, tp 2

Hold order: h1 #38, h3 #46, h4 #46, h6 #38.

| prompt tokens | #38 median `ttft_s` | #46 median `ttft_s` | branch / base | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|---|---|
| 2,048 | 0.406 (6) | 0.413 (6) | 1.015 | h1: 0.395, 0.402, 0.403; h3: 0.404, 0.414, 0.412; h4: 0.403, 0.413, 0.413; h6: 0.410, 0.417, 0.420 | all 0 (16) | adafb7a418699dc9 | equal |
| 8,192 | 1.780 (6) | 1.789 (6) | 1.005 | h1: 1.733, 1.741, 1.745; h3: 1.776, 1.789, 1.789; h4: 1.781, 1.791, 1.794; h6: 1.814, 1.822, 1.831 | all 0 (16) | cd5ab82cd82267ff | equal |
| 32,768 | 9.280 (6) | 9.325 (6) | 1.005 | h1: 8.992, 9.062, 9.119; h3: 9.276, 9.334, 9.317; h4: 9.282, 9.345, 9.332; h6: 9.442, 9.483, 9.450 | all 0 (16) | 132876962ac928f3 | equal |
| 65,536 | 23.879 (6) | 23.813 (6) | 0.997 | h1: 23.697, 23.684, 23.650; h3: 23.809, 23.724, 23.689; h4: 23.847, 23.817, 23.862; h6: 24.063, 24.061, 24.119 | all 0 (16) | 60049d7f13388407 | equal |

### 4 × V620, tp 4


#### Decode, drafts on: 4 × V620, tp 4

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | h1 #38 | h2 #44 | h3 #46 | h4 #46 | h5 #44 | h6 #38 |
|---|---:|---:|---:|---:|---:|---:|
| fibonacci-raw | 196.79 (196.79, 194.36, 199.79) | 230.19 (223.15, 230.19, 230.62) | 239.26 (231.91, 239.26, 239.50) | 232.15 (227.17, 232.53, 232.15) | 226.15 (219.01, 226.15, 227.19) | 197.04 (197.04, 194.44, 197.65) |
| gpu-chat-no-think | 149.74 (142.72, 149.74, 149.83) | 172.36 (165.36, 172.36, 173.70) | 177.39 (172.47, 177.39, 177.73) | 177.21 (167.62, 177.47, 177.21) | 169.96 (162.66, 169.96, 173.59) | 149.62 (149.60, 149.62, 149.67) |

fibonacci-raw: mean of medians #38 196.91, #44 228.17, #46 235.70; #46 / #38 +19.7 %; #44 / #38 +15.9 %. `token_sha` per tree: #38 c91869ce1808; #44 c91869ce1808; #46 c91869ce1808.

gpu-chat-no-think: mean of medians #38 149.68, #44 171.16, #46 177.30; #46 / #38 +18.5 %; #44 / #38 +14.4 %. `token_sha` per tree: #38 d08c6a2623e1; #44 d08c6a2623e1; #46 d08c6a2623e1.

#### Decode, drafts off: 4 × V620, tp 4

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | h1 #38 | h2 #44 | h3 #46 | h4 #46 | h5 #44 | h6 #38 |
|---|---:|---:|---:|---:|---:|---:|
| fibonacci-raw | 107.37 (107.52, 106.91, 107.37) | 115.36 (115.36, 115.21, 115.53) | 118.63 (118.73, 117.69, 118.63) | 119.26 (118.85, 119.46, 119.26) | 115.82 (115.85, 115.82, 115.58) | 107.40 (107.40, 107.39, 107.63) |
| gpu-chat-no-think | 107.47 (107.50, 107.32, 107.47) | 115.42 (115.46, 115.42, 114.98) | 118.86 (119.03, 118.50, 118.86) | 118.71 (118.71, 118.76, 118.21) | 115.66 (115.66, 115.55, 116.33) | 107.26 (107.23, 107.40, 107.26) |

fibonacci-raw: mean of medians #38 107.38, #44 115.59, #46 118.94; #46 / #38 +10.8 %; #44 / #38 +7.6 %. `token_sha` per tree: #38 c91869ce1808; #44 c91869ce1808; #46 c91869ce1808.

gpu-chat-no-think: mean of medians #38 107.36, #44 115.54, #46 118.79; #46 / #38 +10.6 %; #44 / #38 +7.6 %. `token_sha` per tree: #38 d08c6a2623e1; #44 d08c6a2623e1; #46 d08c6a2623e1.

#### Cold prefill, #38 (base) against #46 (branch): 4 × V620, tp 4

Hold order: h1 #38, h3 #46, h4 #46, h6 #38.

| prompt tokens | #38 median `ttft_s` | #46 median `ttft_s` | branch / base | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|---|---|
| 2,048 | 0.512 (6) | 0.510 (6) | 0.995 | h1: 0.518, 0.509, 0.510; h3: 0.518, 0.508, 0.509; h4: 0.520, 0.509, 0.510; h6: 0.522, 0.512, 0.513 | all 0 (16) | c62de829e274093e | equal |
| 8,192 | 2.130 (6) | 2.122 (6) | 0.996 | h1: 2.127, 2.121, 2.121; h3: 2.123, 2.115, 2.118; h4: 2.129, 2.121, 2.124; h6: 2.141, 2.133, 2.135 | all 0 (16) | 51c60772acc9542f | equal |
| 32,768 | 9.903 (6) | 9.880 (6) | 0.998 | h1: 9.853, 9.874, 9.888; h3: 9.853, 9.874, 9.908; h4: 9.874, 9.886, 9.912; h6: 9.918, 9.938, 9.960 | all 0 (16) | 132876962ac928f3 | equal |
| 65,536 | 23.657 (6) | 23.561 (6) | 0.996 | h1: 23.538, 23.588, 23.646; h3: 23.545, 23.549, 23.571; h4: 23.552, 23.571, 23.580; h6: 23.668, 23.695, 23.694 | all 0 (16) | 60049d7f13388407 | equal |

## prefill-chunk


### 1 × V620, tp 1


#### Cold prefill, #37 (base) against #38 (branch): 1 × V620, tp 1

Hold order: h1 #37, h2 #38, h3 #38, h4 #37.

| prompt tokens | #37 median `ttft_s` | #38 median `ttft_s` | branch / base | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|---|---|
| 2,048 | 1.273 (6) | 0.834 (6) | 0.656 | h1: 1.270, 1.272, 1.273; h2: 0.829, 0.835, 0.835; h3: 0.829, 0.835, 0.834; h4: 1.273, 1.275, 1.275 | all 0 (16) | c62de829e274093e | equal |
| 8,192 | 5.344 (6) | 3.567 (6) | 0.667 | h1: 5.330, 5.339, 5.348; h2: 3.566, 3.570, 3.572; h3: 3.560, 3.565, 3.568; h4: 5.342, 5.347, 5.355 | all 0 (16) | 5c418c36f5e1f55d | equal |
| 32,768 | 25.441 (6) | 18.004 (6) | 0.708 | h1: 25.406, 25.451, 25.481; h2: 17.988, 18.009, 18.017; h3: 17.985, 17.999, 18.021; h4: 25.410, 25.431, 25.453 | all 0 (16) | 132876962ac928f3 | equal |
| 65,536 | 61.511 (6) | 45.879 (6) | 0.746 | h1: 61.509, 61.514, 61.523; h2: 45.867, 45.874, 45.857; h3: 45.884, 45.937, 45.918; h4: 61.497, 61.522, 61.470 | all 0 (16) | 60049d7f13388407 | equal |
| 131,072 | 165.194 (6) | 131.203 (6) | 0.794 | h1: 165.193, 165.270, 165.228; h2: 131.121, 131.114, 131.368; h3: 131.174, 131.240, 131.231; h4: 165.195, 165.167, 165.160 | all 0 (16) | 71ac1f6b30a39cba | equal |
| 262,080 | 499.921 (6) | 418.752 (6) | 0.838 | h1: 500.761, 500.192, 500.068; h2: 418.670, 418.836, 419.070; h3: 418.796, 418.708, 418.621; h4: 499.774, 499.759, 499.543 | all 0 (16) | 3af710fa207e3012 | equal |

### 2 × V620, tp 2


#### Cold prefill, #37 (base) against #38 (branch): 2 × V620, tp 2

Hold order: h1 #37, h2 #38, h3 #38, h4 #37.

| prompt tokens | #37 median `ttft_s` | #38 median `ttft_s` | branch / base | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|---|---|
| 2,048 | 0.835 (6) | 0.578 (6) | 0.692 | h1: 0.835, 0.835, 0.838; h2: 0.578, 0.578, 0.580; h3: 0.576, 0.575, 0.578; h4: 0.833, 0.832, 0.835 | all 0 (16) | c62de829e274093e | equal |
| 8,192 | 3.495 (6) | 2.454 (6) | 0.702 | h1: 3.494, 3.498, 3.509; h2: 2.453, 2.455, 2.457; h3: 2.445, 2.448, 2.456; h4: 3.483, 3.488, 3.497 | all 0 (16) | 51c60772acc9542f | equal |
| 32,768 | 16.452 (6) | 12.108 (6) | 0.736 | h1: 16.484, 16.423, 16.421; h2: 12.084, 12.112, 12.133; h3: 12.063, 12.148, 12.104; h4: 16.389, 16.480, 16.493 | all 0 (16) | 132876962ac928f3 | equal |
| 65,536 | 39.016 (6) | 29.932 (6) | 0.767 | h1: 39.076, 38.965, 38.927; h2: 29.987, 29.937, 29.893; h3: 29.992, 29.914, 29.927; h4: 39.022, 39.009, 39.069 | all 0 (16) | 79ee88611a8817b1 | equal |
| 131,072 | 102.461 (6) | 82.564 (6) | 0.806 | h1: 102.527, 102.424, 102.051; h2: 82.881, 82.506, 82.526; h3: 82.848, 82.601, 82.370; h4: 102.607, 102.489, 102.433 | all 0 (16) | 71ac1f6b30a39cba | equal |
| 262,080 | 300.642 (6) | 257.537 (6) | 0.857 | h1: 299.849, 300.562, 303.100; h2: 255.128, 257.860, 258.382; h3: 254.580, 257.214, 258.473; h4: 300.722, 300.475, 303.186 | all 0 (16) | 3af710fa207e3012 | equal |

### 2 × W7800, tp 2


#### Cold prefill, #37 (base) against #38 (branch): 2 × W7800, tp 2

Hold order: h1 #37, h2 #38, h3 #38, h4 #37.

| prompt tokens | #37 median `ttft_s` | #38 median `ttft_s` | branch / base | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|---|---|
| 2,048 | 0.647 (6) | 0.416 (6) | 0.643 | h1: 0.637, 0.645, 0.649; h2: 0.411, 0.417, 0.419; h3: 0.407, 0.416, 0.416; h4: 0.640, 0.648, 0.651 | all 0 (16) | adafb7a418699dc9 | equal |
| 8,192 | 2.753 (6) | 1.808 (6) | 0.657 | h1: 2.730, 2.748, 2.757; h2: 1.803, 1.815, 1.817; h3: 1.798, 1.807, 1.809; h4: 2.742, 2.758, 2.766 | all 0 (16) | cd5ab82cd82267ff | equal |
| 32,768 | 13.310 (6) | 9.416 (6) | 0.707 | h1: 13.297, 13.334, 13.276; h2: 9.389, 9.448, 9.435; h3: 9.356, 9.421, 9.411; h4: 13.325, 13.322, 13.268 | all 0 (16) | 132876962ac928f3 | equal |
| 65,536 | 32.137 (6) | 24.112 (6) | 0.750 | h1: 32.112, 32.150, 32.141; h2: 24.118, 24.094, 24.135; h3: 24.086, 24.106, 24.154; h4: 32.127, 32.134, 32.208 | all 0 (16) | 60049d7f13388407 | equal |
| 131,072 | 87.337 (6) | 69.962 (6) | 0.801 | h1: 87.284, 87.332, 87.343; h2: 69.951, 69.927, 69.915; h3: 70.057, 69.973, 70.109; h4: 87.235, 87.372, 87.351 | all 0 (16) | cfdc90dbc1e14be9 | equal |
| 262,080 | 268.192 (6) | 229.455 (6) | 0.856 | h1: 268.033, 268.225, 267.955; h2: 228.970, 228.964, 229.934; h3: 228.976, 230.606, 230.979; h4: 268.321, 268.299, 268.159 | all 0 (16) | 3af710fa207e3012 | equal |

### 4 × V620, tp 4


#### Cold prefill, #37 (base) against #38 (branch): 4 × V620, tp 4

Hold order: h1 #37, h2 #38, h3 #38, h4 #37.

| prompt tokens | #37 median `ttft_s` | #38 median `ttft_s` | branch / base | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|---|---|
| 2,048 | 0.682 (6) | 0.516 (6) | 0.757 | h1: 0.694, 0.681, 0.682; h2: 0.524, 0.515, 0.516; h3: 0.525, 0.515, 0.516; h4: 0.689, 0.678, 0.679 | all 0 (16) | c62de829e274093e | equal |
| 8,192 | 2.811 (6) | 2.149 (6) | 0.765 | h1: 2.809, 2.805, 2.809; h2: 2.153, 2.144, 2.149; h3: 2.153, 2.144, 2.149; h4: 2.818, 2.812, 2.816 | all 0 (16) | 51c60772acc9542f | equal |
| 32,768 | 12.725 (6) | 9.989 (6) | 0.785 | h1: 12.705, 12.725, 12.756; h2: 9.975, 9.991, 9.997; h3: 9.976, 9.987, 10.000; h4: 12.702, 12.725, 12.735 | all 0 (16) | 132876962ac928f3 | equal |
| 65,536 | 29.268 (6) | 23.745 (6) | 0.811 | h1: 29.280, 29.259, 29.260; h2: 23.738, 23.754, 23.763; h3: 23.724, 23.725, 23.751; h4: 29.273, 29.263, 29.277 | all 0 (16) | 60049d7f13388407 | equal |
| 131,072 | 73.739 (6) | 62.401 (6) | 0.846 | h1: 73.760, 73.775, 73.734; h2: 62.404, 62.400, 62.457; h3: 62.402, 62.381, 62.392; h4: 73.701, 73.650, 73.743 | all 0 (16) | cfdc90dbc1e14be9 | equal |
| 262,080 | 208.635 (6) | 183.676 (6) | 0.880 | h1: 208.678, 208.734, 208.589; h2: 183.826, 183.880, 183.601; h3: 183.714, 183.575, 183.638; h4: 208.486, 208.647, 208.622 | all 0 (16) | 3af710fa207e3012 | equal |

## family


### 1 × V620, tp 1


#### Decode, drafts on: 1 × V620, tp 1

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | f1 #46 |
|---|---:|
| fibonacci-raw | 178.53 (178.41, 178.56, 178.53) |
| gpu-chat-no-think | 155.35 (155.36, 155.35, 155.35) |

fibonacci-raw: mean of medians #46 178.53. `token_sha` per tree: #46 5bb240972d6b.

gpu-chat-no-think: mean of medians #46 155.35. `token_sha` per tree: #46 9f806c1e0e42.

#### `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial`: 1 × V620, tp 1, #46

| prompt | streams | aggregate tok/s | per-stream tok/s | concurrent == alone (equal/unequal/failed) | alone == `"draft": false` |
|---|---:|---:|---:|---|---|
| code | 1 | 178.3 | 178.3 | 3/0/0 | 1/0/0 |
| code | 4 | 254.2 | 66.1 | 12/0/0 | 1/0/0 |
| chat | 1 | 154.9 | 154.9 | 3/0/0 | 1/0/0 |
| chat | 4 | 216.6 | 57.3 | 12/0/0 | 1/0/0 |

#### `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4,8 --alone --serial`: 1 × V620, tp 1, #46

| prompt | streams | aggregate tok/s | per-stream tok/s | concurrent == alone (equal/unequal/failed) | alone == `"draft": false` |
|---|---:|---:|---:|---|---|
| code | 1 | 178.4 | 178.4 | 3/0/0 | 1/0/0 |
| code | 4 | 253.3 | 66.0 | 12/0/0 | 1/0/0 |
| code | 8 | 303.8 | 43.8 | 24/0/0 | 1/0/0 |
| chat | 1 | 154.9 | 154.9 | 3/0/0 | 1/0/0 |
| chat | 4 | 210.4 | 55.1 | 12/0/0 | 1/0/0 |
| chat | 8 | 257.9 | 34.0 | 24/0/0 | 1/0/0 |

#### Cold prefill at #46, 131,072 and 262,080 tokens: 1 × V620, tp 1

Hold order: f2 #46.

| prompt tokens | #46 median `ttft_s` | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|
| 131,072 | 131.067 (3) | f2: 130.793, 131.073, 131.067 | all 0 (4) | 71ac1f6b30a39cba | equal |
| 262,080 | 418.385 (3) | f2: 418.560, 418.182, 418.385 | all 0 (4) | 3af710fa207e3012 | equal |

### 2 × V620, tp 2


#### Decode, drafts on: 2 × V620, tp 2

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | f1 #46 |
|---|---:|
| fibonacci-raw | 206.11 (202.60, 206.11, 211.98) |
| gpu-chat-no-think | 173.20 (171.89, 173.20, 173.49) |

fibonacci-raw: mean of medians #46 206.11. `token_sha` per tree: #46 c91869ce1808.

gpu-chat-no-think: mean of medians #46 173.20. `token_sha` per tree: #46 2fdc4c22dd60.

#### `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial`: 2 × V620, tp 2, #46

| prompt | streams | aggregate tok/s | per-stream tok/s | concurrent == alone (equal/unequal/failed) | alone == `"draft": false` |
|---|---:|---:|---:|---|---|
| code | 1 | 206.2 | 206.2 | 3/0/0 | 1/0/0 |
| code | 4 | 362.6 | 95.8 | 12/0/0 | 1/0/0 |
| chat | 1 | 172.5 | 172.5 | 3/0/0 | 1/0/0 |
| chat | 4 | 301.8 | 79.2 | 12/0/0 | 1/0/0 |

#### `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4,8 --alone --serial`: 2 × V620, tp 2, #46

| prompt | streams | aggregate tok/s | per-stream tok/s | concurrent == alone (equal/unequal/failed) | alone == `"draft": false` |
|---|---:|---:|---:|---|---|
| code | 1 | 211.7 | 211.7 | 3/0/0 | 1/0/0 |
| code | 4 | 363.3 | 95.4 | 12/0/0 | 1/0/0 |
| code | 8 | 431.0 | 64.3 | 24/0/0 | 1/0/0 |
| chat | 1 | 174.6 | 174.6 | 3/0/0 | 1/0/0 |
| chat | 4 | 303.6 | 79.8 | 12/0/0 | 1/0/0 |
| chat | 8 | 363.1 | 51.6 | 24/0/0 | 1/0/0 |

#### Cold prefill at #46, 131,072 and 262,080 tokens: 2 × V620, tp 2

Hold order: f2 #46.

| prompt tokens | #46 median `ttft_s` | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|
| 131,072 | 82.816 (3) | f2: 82.723, 82.933, 82.816 | all 0 (4) | 71ac1f6b30a39cba | equal |
| 262,080 | 256.443 (3) | f2: 256.443, 254.732, 257.266 | all 0 (4) | 3af710fa207e3012 | equal |

### 2 × W7800, tp 2


#### Decode, drafts on: 2 × W7800, tp 2

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | f1 #46 |
|---|---:|
| fibonacci-raw | 243.44 (238.16, 245.16, 243.44) |
| gpu-chat-no-think | 204.67 (201.15, 206.06, 204.67) |

fibonacci-raw: mean of medians #46 243.44. `token_sha` per tree: #46 0808cf694e6f.

gpu-chat-no-think: mean of medians #46 204.67. `token_sha` per tree: #46 25c162db90ad.

#### `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial`: 2 × W7800, tp 2, #46

| prompt | streams | aggregate tok/s | per-stream tok/s | concurrent == alone (equal/unequal/failed) | alone == `"draft": false` |
|---|---:|---:|---:|---|---|
| code | 1 | 244.8 | 244.8 | 3/0/0 | 1/0/0 |
| code | 4 | 427.8 | 111.8 | 12/0/0 | 1/0/0 |
| chat | 1 | 201.9 | 201.9 | 3/0/0 | 1/0/0 |
| chat | 4 | 339.5 | 88.9 | 12/0/0 | 1/0/0 |

#### `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4,8 --alone --serial`: 2 × W7800, tp 2, #46

| prompt | streams | aggregate tok/s | per-stream tok/s | concurrent == alone (equal/unequal/failed) | alone == `"draft": false` |
|---|---:|---:|---:|---|---|
| code | 1 | 241.3 | 241.3 | 3/0/0 | 1/0/0 |
| code | 4 | 410.9 | 106.0 | 12/0/0 | 1/0/0 |
| code | 8 | 467.3 | 70.0 | 24/0/0 | 1/0/0 |
| chat | 1 | 202.1 | 202.1 | 3/0/0 | 1/0/0 |
| chat | 4 | 342.7 | 89.3 | 12/0/0 | 1/0/0 |
| chat | 8 | 404.2 | 56.5 | 24/0/0 | 1/0/0 |

#### Cold prefill at #46, 131,072 and 262,080 tokens: 2 × W7800, tp 2

Hold order: f2 #46.

| prompt tokens | #46 median `ttft_s` | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|
| 131,072 | 69.595 (3) | f2: 69.065, 69.595, 69.667 | all 0 (4) | cfdc90dbc1e14be9 | equal |
| 262,080 | 228.518 (3) | f2: 228.518, 228.419, 228.547 | all 0 (4) | 3af710fa207e3012 | equal |

### 4 × V620, tp 4


#### Decode, drafts on: 4 × V620, tp 4

`bench_openai.py --tokens 256 --reps 3 --temperatures 0`, medians in tok/s (the 3 reps in brackets), holds in order.

| prompt | f1 #46 |
|---|---:|
| fibonacci-raw | 238.58 (231.67, 238.58, 238.64) |
| gpu-chat-no-think | 177.48 (172.51, 177.48, 177.54) |

fibonacci-raw: mean of medians #46 238.58. `token_sha` per tree: #46 c91869ce1808.

gpu-chat-no-think: mean of medians #46 177.48. `token_sha` per tree: #46 d08c6a2623e1.

#### `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial`: 4 × V620, tp 4, #46

| prompt | streams | aggregate tok/s | per-stream tok/s | concurrent == alone (equal/unequal/failed) | alone == `"draft": false` |
|---|---:|---:|---:|---|---|
| code | 1 | 235.3 | 235.3 | 3/0/0 | 1/0/0 |
| code | 4 | 495.0 | 125.0 | 12/0/0 | 1/0/0 |
| chat | 1 | 175.2 | 175.2 | 3/0/0 | 1/0/0 |
| chat | 4 | 369.5 | 96.5 | 12/0/0 | 1/0/0 |

#### `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4,8 --alone --serial`: 4 × V620, tp 4, #46

| prompt | streams | aggregate tok/s | per-stream tok/s | concurrent == alone (equal/unequal/failed) | alone == `"draft": false` |
|---|---:|---:|---:|---|---|
| code | 1 | 239.6 | 239.6 | 3/0/0 | 1/0/0 |
| code | 4 | 487.4 | 124.2 | 12/0/0 | 1/0/0 |
| code | 8 | 506.8 | 78.6 | 24/0/0 | 1/0/0 |
| chat | 1 | 177.4 | 177.4 | 3/0/0 | 1/0/0 |
| chat | 4 | 369.2 | 95.7 | 12/0/0 | 1/0/0 |
| chat | 8 | 426.8 | 60.2 | 24/0/0 | 1/0/0 |

#### Cold prefill at #46, 131,072 and 262,080 tokens: 4 × V620, tp 4

Hold order: f2 #46.

| prompt tokens | #46 median `ttft_s` | every request, in hold order | cached tokens | 16-token reply `token_sha256` | == `"draft": false` |
|---|---|---|---|---|
| 131,072 | 61.993 (3) | f2: 61.765, 61.995, 61.993 | all 0 (4) | cfdc90dbc1e14be9 | equal |
| 262,080 | 183.091 (3) | f2: 182.962, 183.091, 183.269 | all 0 (4) | 3af710fa207e3012 | equal |


## Memory (`family/<rig>/f2_pr46_long`, and the decode servers)

Sampled every 5 s over the 131,072- and 262,080-token holds: peak `VmHWM` and `VmRSS` of each rank's process, and VRAM used per GPU (`rocm-smi`). Host: 503 GiB RAM. Context `--context 262144 --parallel 1 --prompt-cache-gib 0`.

| rig | engine startup line per rank | VRAM used a GPU: before start / after health / peak / after stop (GiB) | peak VmHWM a rank (GiB) | peak sampled VmRSS a rank (GiB) | load to health |
|---|---|---|---|---|---|
| 1 × V620, tp 1 | HIP gfx1030 device 0, rank 0 of 1: Nemotron-H model 17.34 GiB, MTP head loaded; 1 stream at once, 2.02 GiB each at a 262144-token window, of 9.51 GiB left after the model and 2.0 GiB of scratch; prompt cache 0.00 GiB (0 pages, 0 snapshots); prompts in 2048-row chunks; windows of 4 rows | card5: 0.02 / 21.92 / 21.92 / 0.02 of 30 | rank 0 17.72 | rank 0 0.4 | 4.03 s |
| 2 × V620, tp 2 | HIP gfx1030 device 0, rank 0 of 2: Nemotron-H model 9.28 GiB, MTP head loaded; 1 stream at once, 1.14 GiB each at a 262144-token window, of 17.57 GiB left after the model and 2.0 GiB of scratch; prompt cache 0.00 GiB (0 pages, 0 snapshots); prompts in 2048-row chunks; windows of 4 rows<br>HIP gfx1030 device 1, rank 1 of 2: Nemotron-H model 8.27 GiB, MTP head off; 1 stream at once, 1.14 GiB each at a 262144-token window, of 18.58 GiB left after the model and 2.0 GiB of scratch; prompt cache 0.00 GiB (0 pages, 0 snapshots); prompts in 2048-row chunks; windows of 4 rows | card6: 0.02 / 13.32 / 13.34 / 0.02 of 30<br>card7: 0.02 / 11.98 / 12.01 / 0.02 of 30 | rank 0 10.53, rank 1 9.27 | rank 0 1.07, rank 1 0.84 | 7.05 s |
| 2 × W7800, tp 2 | HIP gfx1100 device 0, rank 0 of 2: Nemotron-H model 9.28 GiB, MTP head loaded; 1 stream at once, 1.14 GiB each at a 262144-token window, of 32.15 GiB left after the model and 2.0 GiB of scratch; prompt cache 0.00 GiB (0 pages, 0 snapshots); prompts in 2048-row chunks; windows of 4 rows<br>HIP gfx1100 device 1, rank 1 of 2: Nemotron-H model 8.27 GiB, MTP head off; 1 stream at once, 1.14 GiB each at a 262144-token window, of 33.17 GiB left after the model and 2.0 GiB of scratch; prompt cache 0.00 GiB (0 pages, 0 snapshots); prompts in 2048-row chunks; windows of 4 rows | card9: 0.03 / 13.38 / 13.41 / 0.03 of 45<br>card8: 0.03 / 12.05 / 12.08 / 0.03 of 45 | rank 0 10.61, rank 1 9.35 | rank 0 1.11, rank 1 0.92 | 7.05 s |
| 4 × V620, tp 4 | HIP gfx1030 device 0, rank 0 of 4: Nemotron-H model 5.26 GiB, MTP head loaded; 1 stream at once, 1.07 GiB each at a 262144-token window, of 21.60 GiB left after the model and 2.0 GiB of scratch; prompt cache 0.00 GiB (0 pages, 0 snapshots); prompts in 2048-row chunks; windows of 4 rows<br>HIP gfx1030 device 1, rank 1 of 4: Nemotron-H model 4.24 GiB, MTP head off; 1 stream at once, 1.07 GiB each at a 262144-token window, of 22.61 GiB left after the model and 2.0 GiB of scratch; prompt cache 0.00 GiB (0 pages, 0 snapshots); prompts in 2048-row chunks; windows of 4 rows<br>HIP gfx1030 device 2, rank 2 of 4: Nemotron-H model 4.24 GiB, MTP head off; 1 stream at once, 1.07 GiB each at a 262144-token window, of 22.61 GiB left after the model and 2.0 GiB of scratch; prompt cache 0.00 GiB (0 pages, 0 snapshots); prompts in 2048-row chunks; windows of 4 rows<br>HIP gfx1030 device 3, rank 3 of 4: Nemotron-H model 4.24 GiB, MTP head off; 1 stream at once, 1.07 GiB each at a 262144-token window, of 22.61 GiB left after the model and 2.0 GiB of scratch; prompt cache 0.00 GiB (0 pages, 0 snapshots); prompts in 2048-row chunks; windows of 4 rows | card4: 0.02 / 9.24 / 9.27 / 0.02 of 30<br>card5: 0.02 / 7.93 / 7.96 / 0.02 of 30<br>card6: 0.02 / 7.93 / 7.96 / 0.02 of 30<br>card7: 0.02 / 7.91 / 7.95 / 0.02 of 30 | rank 0 6.6, rank 1 5.31, rank 2 5.31, rank 3 5.29 | rank 0 1.07, rank 1 0.84, rank 2 0.84, rank 3 0.84 | 9.06 s |

Decode servers (`--context 65536 --parallel 4`, default prompt cache; `--context 32768 --parallel 8` for `conc8`), VRAM used a GPU after health and after the work, GiB:

| rig | mode | after health | after the work | after stop | load to health |
|---|---|---|---|---|---|
| 1 × V620, tp 1 | decode | card5 29.03 | card5 29.04 | card5 0.02 | 4.03 s |
| 1 × V620, tp 1 | conc | card5 29.03 | card5 29.04 | card5 0.02 | 4.03 s |
| 1 × V620, tp 1 | conc8 | card5 29.05 | card5 29.06 | card5 0.02 | 4.03 s |
| 2 × V620, tp 2 | decode | card6 29.54, card7 24.25 | card6 29.60, card7 24.31 | card6 0.02, card7 0.02 | 7.04 s |
| 2 × V620, tp 2 | conc | card6 29.54, card7 24.25 | card6 29.70, card7 24.40 | card6 0.02, card7 0.02 | 7.05 s |
| 2 × V620, tp 2 | conc8 | card6 29.56, card7 24.41 | card6 29.78, card7 24.63 | card6 0.02, card7 0.02 | 7.04 s |
| 2 × W7800, tp 2 | decode | card9 44.20, card8 35.26 | card9 44.26, card8 35.32 | card9 0.03, card8 0.03 | 9.07 s |
| 2 × W7800, tp 2 | conc | card9 44.20, card8 35.26 | card9 44.35, card8 35.41 | card9 0.03, card8 0.03 | 9.06 s |
| 2 × W7800, tp 2 | conc8 | card9 44.19, card8 35.39 | card9 44.43, card8 35.63 | card9 0.03, card8 0.03 | 7.05 s |
| 4 × V620, tp 4 | decode | card4 29.67, card5 23.30, card6 23.30, card7 23.29 | card4 29.73, card5 23.37, card6 23.37, card7 23.36 | card4 0.02, card5 0.02, card6 0.02, card7 0.02 | 9.06 s |
| 4 × V620, tp 4 | conc | card4 29.67, card5 23.30, card6 23.30, card7 23.29 | card4 29.79, card5 23.42, card6 23.42, card7 23.41 | card4 0.02, card5 0.02, card6 0.02, card7 0.02 | 9.06 s |
| 4 × V620, tp 4 | conc8 | card4 29.67, card5 23.37, card6 23.37, card7 23.36 | card4 29.83, card5 23.53, card6 23.53, card7 23.52 | card4 0.02, card5 0.02, card6 0.02, card7 0.02 | 9.06 s |

## Fidelity (`fidelity/`)

Teacher-forced top-1 agreement with the fp32 reference, 480 pooled rows a rig and path:

| rig and path | top-1 | max \|error\| | mean \|error\| |
|---|---:|---:|---:|
| v620-tp1 decode | 479/480 | 2.418 | 0.0360 |
| v620-tp1 prompt | 480/480 | 2.476 | 0.0292 |
| v620-tp2 decode | 479/480 | 2.502 | 0.0326 |
| v620-tp2 prompt | 480/480 | 3.785 | 0.0452 |
| v620-tp4 decode | 479/480 | 2.506 | 0.0356 |
| v620-tp4 prompt | 479/480 | 2.262 | 0.0352 |
| w7800-tp1 decode | 475/480 | 5.754 | 0.1043 |
| w7800-tp1 prompt | 474/480 | 6.129 | 0.1226 |
| w7800-tp2 decode | 477/480 | 6.191 | 0.1053 |
| w7800-tp2 prompt | 476/480 | 4.412 | 0.1103 |

Every top-token difference. Reference margin: the reference's logit of its top token minus its logit of the native top token. Native margin: the native logit of its top token minus its logit of the reference's top token (0.000: the two logits are equal, and the argmax takes the lower id). Position: the row's index in the sequence (the row predicts the token after it).

| rig | path | sequence | position | reference top | native top | reference margin | native margin |
|---|---|---|---:|---|---|---:|---:|
| v620-tp1 | decode | chat_long | 60 | 117316 `' Constraint'` | 83666 `' Constra'` | 0.081 | 0.000 |
| v620-tp2 | decode | chat_long | 60 | 117316 `' Constraint'` | 83666 `' Constra'` | 0.081 | 0.000 |
| v620-tp4 | decode | chat_long | 119 | 1321 `' and'` | 1319 `' ('` | 0.606 | 0.031 |
| v620-tp4 | prompt | chat_long | 60 | 117316 `' Constraint'` | 83666 `' Constra'` | 0.081 | 0.375 |
| w7800-tp1 | decode | story_full | 46 | 1514 `' he'` | 1261 `' a'` | 0.050 | 0.000 |
| w7800-tp1 | decode | chat_long | 60 | 117316 `' Constraint'` | 83666 `' Constra'` | 0.081 | 0.000 |
| w7800-tp1 | decode | chat_long | 76 | 4342 `').\n'` | 1394 `' for'` | 1.062 | 0.625 |
| w7800-tp1 | decode | chat_long | 222 | 1278 `' the'` | 2147 `' other'` | 0.503 | 0.125 |
| w7800-tp1 | decode | code_long | 81 | 19639 `' """'` | 2181 `' data'` | 0.071 | 0.000 |
| w7800-tp1 | prompt | story_full | 21 | 1321 `' and'` | 2274 `' who'` | 0.133 | 0.125 |
| w7800-tp1 | prompt | story_full | 46 | 1514 `' he'` | 1261 `' a'` | 0.050 | 0.000 |
| w7800-tp1 | prompt | chat_long | 60 | 117316 `' Constraint'` | 83666 `' Constra'` | 0.081 | 0.000 |
| w7800-tp1 | prompt | chat_long | 119 | 1321 `' and'` | 1319 `' ('` | 0.606 | 0.750 |
| w7800-tp1 | prompt | chat_long | 243 | 1010 `'\n'` | 1050 `'2'` | 0.594 | 0.250 |
| w7800-tp1 | prompt | code_long | 81 | 19639 `' """'` | 2181 `' data'` | 0.071 | 0.125 |
| w7800-tp2 | decode | chat_long | 60 | 117316 `' Constraint'` | 83666 `' Constra'` | 0.081 | 0.000 |
| w7800-tp2 | decode | chat_long | 76 | 4342 `').\n'` | 1394 `' for'` | 1.062 | 0.125 |
| w7800-tp2 | decode | chat_long | 222 | 1278 `' the'` | 2147 `' other'` | 0.503 | 0.500 |
| w7800-tp2 | prompt | story_full | 21 | 1321 `' and'` | 2274 `' who'` | 0.133 | 0.062 |
| w7800-tp2 | prompt | chat_long | 60 | 117316 `' Constraint'` | 83666 `' Constra'` | 0.081 | 0.500 |
| w7800-tp2 | prompt | chat_long | 222 | 1278 `' the'` | 2147 `' other'` | 0.503 | 0.125 |
| w7800-tp2 | prompt | code_long | 81 | 19639 `' """'` | 2181 `' data'` | 0.071 | 0.000 |
