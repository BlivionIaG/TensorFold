//! Test fixtures: the planner's geometry fields from two Qwen3.5-family config.json files (an A3B MoE and a 4B dense).

pub const a3b_json =
    \\{ "architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
    \\  "quantization": { "group_size": 64, "bits": 4, "mode": "affine",
    \\    "language_model.model.layers.0.mlp.gate": { "group_size": 64, "bits": 8 },
    \\    "language_model.model.layers.0.mlp.shared_expert_gate": { "group_size": 64, "bits": 8 } },
    \\  "text_config": { "model_type": "qwen3_5_moe_text", "hidden_size": 2048, "num_hidden_layers": 40,
    \\    "layer_types": [
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention", "full_attention" ],
    \\    "full_attention_interval": 4, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
    \\    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    \\    "linear_num_value_heads": 32, "linear_value_head_dim": 128,
    \\    "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
    \\    "shared_expert_intermediate_size": 512, "vocab_size": 248320, "tie_word_embeddings": false } }
;

pub const d4_json =
    \\{ "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
    \\  "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
    \\  "text_config": { "model_type": "qwen3_5_text", "hidden_size": 2560, "num_hidden_layers": 32,
    \\    "full_attention_interval": 4, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
    \\    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    \\    "linear_num_value_heads": 32, "linear_value_head_dim": 128,
    \\    "intermediate_size": 9216, "vocab_size": 248320, "tie_word_embeddings": true } }
;
