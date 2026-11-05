"""Supervised fine-tuning data: chat-formatted prompt + JSON target, loss on the target only.

Training objective: standard causal-LM cross-entropy, but prompt tokens get
label -100 so the model is only trained to produce
    {"label": ..., "evidence": [...]}<|im_end|>
given the (system, user) messages. This is "completion-only" SFT.
"""
from __future__ import annotations

from .prompting import build_messages

IGNORE_INDEX = -100


def encode_example(record: dict, tok, max_len: int) -> dict | None:
    prompt = tok.apply_chat_template(build_messages(record), tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(build_messages(record, with_target=True), tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    completion = full[len(prompt):]          # target JSON + end-of-turn token(s)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(c_ids) > max_len:
        return None                          # counted and reported by the caller
    return {"input_ids": p_ids + c_ids,
            "labels": [IGNORE_INDEX] * len(p_ids) + c_ids,
            "n_prompt": len(p_ids), "n_target": len(c_ids)}


def encode_split(records, tok, max_len):
    out, skipped = [], []
    for r in records:
        e = encode_example(r, tok, max_len)
        (skipped.append(r["annotation_id"]) if e is None else out.append(e))
    return out, skipped


class ListDataset:
    """Map-style dataset (anything with __len__/__getitem__ works with DataLoader)."""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id: int):
    """Right-pads a batch; padded label positions are ignored by the loss."""
    import torch

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), IGNORE_INDEX, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}
    return collate
