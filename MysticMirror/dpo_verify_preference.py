# -*- coding: utf-8 -*-
"""DPO 效果验证：对比基座模型与 DPO 模型对 chosen/rejected 的偏好。

用同一批偏好数据分别计算 base(SFT) 与 dpo 模型对 chosen/rejected 的
序列 logps，统计 reward acc = P(logp_chosen > logp_rejected)。
预期：DPO 后 acc 明显高于基座。
"""
import json
import sys

import torch

sys.path.insert(0, r"E:\MiniChat\MysticMirror")

from modeling_mystic_mirror import MysticMirrorConfig, MysticMirrorForCausalLM
from train_dpo import DPODataset, compute_sequence_logps
from transformers import AutoConfig, AutoTokenizer

BASE_DIR = r"E:\MiniChat\MysticMirror\smoke_test\sft\final_sft"
DPO_DIR = r"E:\MiniChat\MysticMirror\smoke_test\dpo\best_dpo"
DATA_PATH = r"E:\MiniChat\MysticMirror\data\dpo_smoke.jsonl"

AutoConfig.register("MysticMirror", MysticMirrorConfig)


def load_pair(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = MysticMirrorForCausalLM.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def eval_acc(model, tokenizer, ds):
    accs = []
    logp_gaps = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            ids = s["chosen_input_ids"].unsqueeze(0)
            lab = s["chosen_labels"].unsqueeze(0)
            msk = s["chosen_attention_mask"].unsqueeze(0)
            logp_c = compute_sequence_logps(model, ids, lab, msk)
            ids = s["rejected_input_ids"].unsqueeze(0)
            lab = s["rejected_labels"].unsqueeze(0)
            msk = s["rejected_attention_mask"].unsqueeze(0)
            logp_r = compute_sequence_logps(model, ids, lab, msk)
            gap = (logp_c - logp_r).item()
            logp_gaps.append(gap)
            accs.append(1.0 if gap > 0 else 0.0)
    return sum(accs) / len(accs), sum(logp_gaps) / len(logp_gaps)


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    base_model, tok = load_pair(BASE_DIR)
    dpo_model, _ = load_pair(DPO_DIR)
    ds = DPODataset(data, tok, max_seq_len=512)

    acc_base, gap_base = eval_acc(base_model, tok, ds)
    acc_dpo, gap_dpo = eval_acc(dpo_model, tok, ds)
    print(f"样本数: {len(ds)}")
    print(f"SFT 基座  : reward acc={acc_base:.3f} | 平均 logp gap(chosen-rejected)={gap_base:.4f}")
    print(f"DPO 模型  : reward acc={acc_dpo:.3f} | 平均 logp gap(chosen-rejected)={gap_dpo:.4f}")
    assert acc_dpo > acc_base, "DPO 后偏好应优于基座"
    assert gap_dpo > gap_base, "DPO 后 logp gap 应增大"
    print("✓ DPO 训练确实提升了模型对 chosen 的偏好")


if __name__ == "__main__":
    main()
