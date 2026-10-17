"""Target model configurations for benchmarking."""
from dataclasses import dataclass
from typing import List


@dataclass
class ModelSpec:
    name: str
    n_layer: int
    n_expert: int
    d_model: int
    d_ff: int
    moe_intermediate_size: int
    shared_expert_size: int
    top_k: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    attention_types: List[str]  # per-layer attention type
    mlp_types: List[str]  # 'dense' or 'sparse' (MoE)
    has_mtp: bool = False
    mtp_layers: int = 0
    linear_attention: bool = True
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    first_k_dense: int = 0


GLM_5_3 = ModelSpec(
    name="GLM-5.3",
    n_layer=45,
    n_expert=288,
    d_model=6144,
    d_ff=12288,
    moe_intermediate_size=2048,
    shared_expert_size=12288,
    top_k=8,
    n_heads=48,
    n_kv_heads=48,
    head_dim=128,
    vocab_size=154880,
    attention_types=["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"] + 
                    ["linear_attention"]*3 + ["deepseek_sparse_attention"],
    mlp_types=["dense"]*3 + ["sparse"]*42,
    has_mtp=False,
    linear_attention=True,
    q_lora_rank=1536,
    kv_lora_rank=512,
    first_k_dense=3,
)

GLM_5_3_FLASH = ModelSpec(
    name="GLM-5.3-Flash",
    n_layer=45,
    n_expert=288,
    d_model=4096,
    d_ff=12288,
    moe_intermediate_size=2048,
    shared_expert_size=12288,
    top_k=8,
    n_heads=64,
    n_kv_heads=64,
    head_dim=128,
    vocab_size=154880,
    attention_types=["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"],
    mlp_types=["dense"]*3 + ["sparse"]*42,
    has_mtp=False,
    linear_attention=True,
    q_lora_rank=1536,
    kv_lora_rank=512,
    first_k_dense=3,
)

QWEN_3_8_FLASH_NEXT = ModelSpec(
    name="Qwen3.8-Flash-Next",
    n_layer=48,
    n_expert=512,
    d_model=2560,
    d_ff=640,
    moe_intermediate_size=640,
    shared_expert_size=640,
    top_k=10,
    n_heads=24,
    n_kv_heads=2,
    head_dim=256,
    vocab_size=248320,
    attention_types=["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"] + 
                    ["linear_attention"]*3 + ["full_attention"],
    mlp_types=["dense"]*0 + ["sparse"]*48,  # all sparse
    has_mtp=True,
    mtp_layers=1,
    linear_attention=True,
    q_lora_rank=0,
    kv_lora_rank=0,
    first_k_dense=0,
)


def get_spec(name: str) -> ModelSpec:
    specs = {
        "glm-5.3": GLM_5_3,
        "glm-5.3-flash": GLM_5_3_FLASH,
        "qwen3.8-flash-next": QWEN_3_8_FLASH_NEXT,
    }
    return specs[name.lower()]
