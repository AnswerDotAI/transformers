import torch
from transformers import AutoConfig, AutoModelForCausalLM


if __name__ == "__main__":
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg = AutoConfig.from_pretrained(model)
    cfg._attn_implementation = "eager"
    cfg.use_fp8_kv_scale = True
    cfg.cla_kv_cache_map = {k: k//2 for k in range(24)}
    cfg.cla_shared_coef = 0.5
    cfg.palu_kv_compression_enabled = False
    cfg.use_cache = False
    cfg.debug_kv_sharing = True
    cfg.output_attentions = True

    model = AutoModelForCausalLM.from_config(cfg)
    model.to(device="cuda", dtype=torch.bfloat16)
    
    x = torch.arange(32, device="cuda").view(1,-1)
    out = model(x)
    print(out.attentions[0])
