
        auto n = thread_position_in_grid.z;                 // (stream, head)
        const int st = int(n) / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        const device float* Sb = S0;
        switch (st) {
          case 1: Sb = S1; break;
          case 2: Sb = S2; break;
          case 3: Sb = S3; break;
          case 4: Sb = S4; break;
          case 5: Sb = S5; break;
          case 6: Sb = S6; break;
          case 7: Sb = S7; break;
        }
        auto i_state = Sb + (hv_idx * Dv + dv_idx) * Dk;
        float s0[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          s0[i] = static_cast<float>(i_state[s_idx]);
        }
        float states[MAXW][n_per_t];
        const int first = meta[2 * st], W = meta[2 * st + 1];   // the stream's rows among all rows
        for (int node = 0; node < W; ++node) {
          const int parent = parents[first + node];
          float state[n_per_t];
          // a chain keeps one slot: each node's parent is the node before it
          for (int i = 0; i < n_per_t; ++i) state[i] = parent < 0 ? s0[i] : states[CHAIN ? 0 : parent][i];
          const int row = first + node;
          auto q_ = q + (row * Hk + hk_idx) * Dk;
          auto k_ = k + (row * Hk + hk_idx) * Dk;
          auto v_ = v + (row * Hv + hv_idx) * Dv;
          const float g_ = static_cast<float>(g[row * Hv + hv_idx]);
          const float beta_ = static_cast<float>(beta[row * Hv + hv_idx]);
          // --- mlx_lm gated_delta_step, one step, verbatim arithmetic ---
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_;
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (v_[dv_idx] - kv_mem) * beta_;
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[(row * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }
          for (int i = 0; i < n_per_t; ++i) states[CHAIN ? 0 : node][i] = state[i];
        }

        device float* Ob = O0;
        switch (st) {
          case 1: Ob = O1; break;
          case 2: Ob = O2; break;
          case 3: Ob = O3; break;
          case 4: Ob = O4; break;
          case 5: Ob = O5; break;
          case 6: Ob = O6; break;
          case 7: Ob = O7; break;
        }
        auto o_state = Ob + (hv_idx * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = states[0][i];
