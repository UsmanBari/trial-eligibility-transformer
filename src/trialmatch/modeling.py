"""Model / tokenizer loading shared by evaluation, training and the app."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device_dtype():
    if torch.cuda.is_available():
        # T4/P100 have no bf16 support -> fp16; Ampere+ -> bf16.
        major, _ = torch.cuda.get_device_capability()
        return "cuda", (torch.bfloat16 if major >= 8 else torch.float16)
    return "cpu", torch.float32


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # left padding for batched generation
    return tok


def load_model(name: str, adapter: str | None = None, device=None, dtype=None, merge=True):
    d, t = pick_device_dtype()
    device, dtype = device or d, dtype or t
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.to(device).eval()
    return model


# Pure greedy decoding. Qwen2.5-Instruct's generation_config ships sampling settings AND
# repetition_penalty>1, which generate() still applies when do_sample=False; we switch all
# of it off so generate(), our KV-cache decoder and speculative decoding are comparable.
GREEDY_KWARGS = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)


def model_card(model, name: str) -> dict:
    cfg = model.config
    return {
        "name": name,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "n_heads": getattr(cfg, "num_attention_heads", None),
        "n_kv_heads": getattr(cfg, "num_key_value_heads", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "dtype": str(next(model.parameters()).dtype),
    }
