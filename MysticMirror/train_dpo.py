# -*- coding: utf-8 -*-
"""
用法示例：
    python train_dpo.py --base_model_dir ./sft_output/best_sft \
        --data_path ./data/dpo.jsonl --output_dir ./dpo_output \
        --beta 0.1 --learning_rate 1e-6 --num_train_epochs 1
"""
import argparse
import json
import math
import os
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from typing import Dict, List, Optional

from modeling_mystic_mirror import MysticMirrorConfig, MysticMirrorForCausalLM
from train_sft import (
    print_trainable_parameters,
    create_optimizer_and_scheduler,
    load_json_or_jsonl,
    split_data,
)

AutoConfig.register("MysticMirror", MysticMirrorConfig)
AutoModelForCausalLM.register(MysticMirrorConfig, MysticMirrorForCausalLM)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# ============================================================
# 核心组件：序列对数概率 / DPO 损失 / 数据 / collate
# ============================================================
def compute_sequence_logps(
    model, input_ids, labels, attention_mask
) -> torch.Tensor:
    """计算每个样本"有效监督位置"的序列对数概率（长度归一化均值）。

    - 自回归语义：位置 t 的 logits 预测位置 t+1 的目标 token，先 shift；
    - labels 为 -100 的位置不参与统计（prompt / padding）；
    - 返回形状 (batch,)，数值为负。
    """
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
    per_token_logps = torch.gather(
        log_probs, -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)  # (b, seq-1)

    valid = (shift_labels != -100).float()
    if attention_mask is not None:
        valid = valid * attention_mask[:, 1:].float()
    seq_logps = (per_token_logps * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
    return seq_logps


def dpo_loss(
    pi_chosen,
    pi_rejected,
    ref_chosen,
    ref_rejected,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
):
    """标准 DPO 损失（Bradley-Terry 对数比值）。

    Args:
        pi_chosen / pi_rejected: policy 对 chosen/rejected 的序列 logps
        ref_chosen / ref_rejected: reference 的序列 logps
        beta: 温度系数，越大越强调与参考模型的偏差
        label_smoothing: 标签平滑（0 或 0.1）

    Returns:
        (loss 标量 tensor, reward 准确率 float)
    """
    pi_logratios = pi_chosen - pi_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = beta * (pi_logratios - ref_logratios)

    if label_smoothing > 0:
        loss = (1 - label_smoothing) * -F.logsigmoid(logits) \
            - label_smoothing * F.logsigmoid(-logits)
    else:
        loss = -F.logsigmoid(logits)

    with torch.no_grad():
        acc = torch.sigmoid(logits).mean().item()
    return loss.mean(), acc


def _render_text_pair(tokenizer, prompt, response, max_seq_len):
    """渲染 prompt + response 文本对为 (input_ids, labels)。

    user 部分（含 <|im_end|>）与 assistant 前缀：labels=-100；
    assistant 回复内容与 <|im_end|>：labels=真实 token（计损失）。
    """
    user_text = f"{IM_START}user\n{prompt}{IM_END}\n"
    asst_prefix = f"{IM_START}assistant\n"
    body = f"{response}{IM_END}\n"

    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    prefix_ids = tokenizer.encode(asst_prefix, add_special_tokens=False)
    body_ids = tokenizer.encode(body, add_special_tokens=False)

    input_ids = user_ids + prefix_ids + body_ids
    labels = [-100] * (len(user_ids) + len(prefix_ids)) + body_ids

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
    if all(l == -100 for l in labels):
        return None
    return input_ids, labels


def _render_messages(tokenizer, messages, max_seq_len):
    """按 SFT 规则渲染消息列表：system/user/tool 为 -100，assistant 计损失。

    与 train_sft.SFTDataset._format_example 的模板逐字一致。
    """
    input_ids: List[int] = []
    labels: List[int] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "assistant":
            prefix = f"{IM_START}assistant\n"
            body = f"{content}{IM_END}\n"
        elif role == "tool":
            prefix = f"{IM_START}user\n<tool_response>\n"
            body = f"{content}\n</tool_response>{IM_END}\n"
        else:  # system / user
            prefix = f"{IM_START}{role}\n{content}{IM_END}\n"
            body = ""
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        input_ids.extend(prefix_ids)
        labels.extend([-100] * len(prefix_ids))
        if body:
            body_ids = tokenizer.encode(body, add_special_tokens=False)
            input_ids.extend(body_ids)
            labels.extend(body_ids if role == "assistant" else [-100] * len(body_ids))
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
    if all(l == -100 for l in labels):
        return None
    return input_ids, labels


class DPODataset(Dataset):
    """DPO 偏好数据集。

    输出字段：chosen_input_ids / chosen_labels / chosen_attention_mask 与
    rejected_* 一一对应。chosen 与 rejected 各自独立渲染与截断。
    """

    def __init__(self, data_list, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if tokenizer.eos_token_id is None:
            raise ValueError("分词器缺少 eos_token_id！")
        self.samples: List[Dict[str, torch.Tensor]] = []
        self._build(data_list)

    @staticmethod
    def _extract_pair(item):
        chosen = item.get("chosen", item.get("choose"))
        rejected = item.get("rejected", item.get("reject"))
        prompt = item.get("prompt", "") or ""
        if chosen is None or rejected is None:
            return None
        return chosen, rejected, prompt

    def _format(self, item) -> Optional[Dict[str, torch.Tensor]]:
        pair = self._extract_pair(item)
        if pair is None:
            warnings.warn(f"缺少 chosen/rejected 字段，跳过: {list(item.keys())}")
            return None
        chosen, rejected, prompt = pair

        if isinstance(chosen, str) and isinstance(rejected, str):
            if not prompt:
                warnings.warn("文本格式但缺少 prompt 字段，跳过")
                return None
            c = _render_text_pair(self.tokenizer, prompt, chosen, self.max_seq_len)
            r = _render_text_pair(self.tokenizer, prompt, rejected, self.max_seq_len)
            if c is None or r is None:
                return None
            c_ids, c_labels = c
            r_ids, r_labels = r

        elif isinstance(chosen, list) and isinstance(rejected, list):
            if prompt:
                chosen = [{"role": "user", "content": prompt}] + chosen
                rejected = [{"role": "user", "content": prompt}] + rejected
            c = _render_messages(self.tokenizer, chosen, self.max_seq_len)
            r = _render_messages(self.tokenizer, rejected, self.max_seq_len)
            if c is None or r is None:
                return None
            c_ids, c_labels = c
            r_ids, r_labels = r
        else:
            warnings.warn(f"chosen/rejected 类型不支持（{type(chosen).__name__}/"
                          f"{type(rejected).__name__}），跳过")
            return None

        return {
            "chosen_input_ids": torch.tensor(c_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(c_labels, dtype=torch.long),
            "chosen_attention_mask": torch.ones(len(c_ids), dtype=torch.long),
            "rejected_input_ids": torch.tensor(r_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(r_labels, dtype=torch.long),
            "rejected_attention_mask": torch.ones(len(r_ids), dtype=torch.long),
        }

    def _build(self, data_list):
        for item in tqdm(data_list, desc="Building DPO samples"):
            sample = self._format(item)
            if sample is not None:
                self.samples.append(sample)
        if len(self.samples) == 0:
            warnings.warn("DPO 数据集为空！请检查数据格式。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def dpo_collate_fn(batch, pad_token_id: int):
    """chosen / rejected 各自右侧 padding 到 batch 内最大长度。"""

    def _pad(prefix):
        ids = [x[f"{prefix}_input_ids"] for x in batch]
        labels = [x[f"{prefix}_labels"] for x in batch]
        max_len = max(len(t) for t in ids)
        out_ids, out_labels, out_mask = [], [], []
        for i, seq_ids in enumerate(ids):
            pad_len = max_len - len(seq_ids)
            out_ids.append(torch.cat([
                seq_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
            out_labels.append(torch.cat([
                labels[i], torch.full((pad_len,), -100, dtype=torch.long)]))
            out_mask.append(torch.cat([
                torch.ones(len(seq_ids), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)]))
        return (torch.stack(out_ids), torch.stack(out_labels),
                torch.stack(out_mask))

    ci, cl, cm = _pad("chosen")
    ri, rl, rm = _pad("rejected")
    return {
        "chosen_input_ids": ci,
        "chosen_labels": cl,
        "chosen_attention_mask": cm,
        "rejected_input_ids": ri,
        "rejected_labels": rl,
        "rejected_attention_mask": rm,
    }


# ============================================================
# DPO 训练主函数
# ============================================================
def train_dpo(
    base_model_dir: str,
    data_path: str,
    output_dir: str = "./dpo_output",
    max_seq_len: int = 2048,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-6,
    weight_decay: float = 0.0,
    warmup_steps: int = 100,
    num_train_epochs: int = 1,
    max_grad_norm: float = 1.0,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    eval_steps: int = 100,
    save_steps: int = 500,
    seed: int = 42,
    mixed_precision: str = "bf16",
    resume_from_checkpoint: Optional[str] = None,
):
    """MysticMirror DPO 训练主入口。

    Args:
        base_model_dir: 基座模型目录（SFT 后的模型效果更好）。
        data_path: DPO 数据文件（json / jsonl）。
        output_dir: 输出目录。
        max_seq_len: 最大序列长度。
        batch_size: 每张卡 batch size。
        gradient_accumulation_steps: 梯度累积步数。
        learning_rate: 学习率（DPO 通常比 SFT 再小 10x）。
        weight_decay: weight decay。
        warmup_steps: 预热步数。
        num_train_epochs: 训练轮数。
        max_grad_norm: 梯度裁剪。
        beta: DPO 温度系数。
        label_smoothing: DPO 标签平滑。
        eval_steps: 评估间隔。
        save_steps: 保存 checkpoint 间隔。
        seed: 随机种子。
        mixed_precision: 混合精度。
        resume_from_checkpoint: 恢复训练的 checkpoint 目录（形如
            checkpoint-2000）；不填则从头训练。
    """
    if not os.path.isdir(base_model_dir):
        raise FileNotFoundError(
            f"基座模型目录不存在: {base_model_dir}\n"
            "请先完成 SFT 训练（生成 ./sft_output/best_sft），"
            "或通过 --base_model_dir 指定已有的模型目录。"
        )
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"DPO 数据文件不存在: {data_path}\n"
            "请用 python download_data.py --files dpo 下载，"
            "或检查 --data_path 是否正确。"
        )
    if resume_from_checkpoint is not None and not os.path.isdir(resume_from_checkpoint):
        raise FileNotFoundError(f"恢复目录不存在: {resume_from_checkpoint}")
    if beta <= 0:
        raise ValueError(f"beta 必须 > 0，得到 {beta}")
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"label_smoothing 必须在 [0, 1)，得到 {label_smoothing}")

    set_seed(seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_dir=output_dir,
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"基座模型: {base_model_dir}")
        print(f"DPO 数据: {data_path}")
        print(f"输出目录: {output_dir} | beta={beta}")

    # ---------- 1. 加载分词器、policy 与 reference ----------
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    torch_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float32
    policy = MysticMirrorForCausalLM.from_pretrained(
        base_model_dir, torch_dtype=torch_dtype
    )
    ref_model = MysticMirrorForCausalLM.from_pretrained(
        base_model_dir, torch_dtype=torch_dtype
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    if accelerator.is_main_process:
        print_trainable_parameters(policy)
        print("reference model 已冻结（不参与反向传播）。")

    # ---------- 2. 数据 ----------
    data_list = load_json_or_jsonl(data_path)
    if not data_list:
        raise ValueError(f"数据文件为空或格式无法解析: {data_path}")
    train_data, eval_data = split_data(data_list, eval_ratio=0.05, seed=seed)
    train_dataset = DPODataset(train_data, tokenizer, max_seq_len)
    eval_dataset = DPODataset(eval_data, tokenizer, max_seq_len)
    if len(train_dataset) == 0:
        raise ValueError(
            f"训练样本为空（原始 {len(data_list)} 条，0 条可用）！\n"
            "请检查数据格式：需要 prompt+chosen+rejected（或 choose/reject）"
            "文本对，或 chosen/rejected 消息列表；并确认 max_seq_len 未截断全部样本。"
        )
    if accelerator.is_main_process:
        print(f"原始数据: {len(data_list)} 条 | 训练样本: {len(train_dataset)}, "
              f"评估样本: {len(eval_dataset)}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    num_workers = 0 if os.name == "nt" else 4
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda b: dpo_collate_fn(b, pad_id),
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda b: dpo_collate_fn(b, pad_id),
    )

    # ---------- 3. 优化器 ----------
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch * num_train_epochs
    warmup = min(warmup_steps, int(num_train_steps * 0.05))
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        policy, num_train_steps, learning_rate, weight_decay, warmup
    )

    # ---------- 4. prepare（policy 训练、ref 冻结） ----------
    policy, ref_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            policy, ref_model, optimizer,
            train_dataloader, eval_dataloader, lr_scheduler,
        )
    )

    global_step = 0
    if resume_from_checkpoint is not None:
        accelerator.load_state(resume_from_checkpoint)
        try:
            global_step = int(os.path.basename(resume_from_checkpoint).split("-")[-1])
        except (ValueError, IndexError):
            warnings.warn(
                f"无法从目录名解析 global_step（{resume_from_checkpoint}），"
                "从 0 开始记录步数；优化器/调度器状态已恢复。"
            )
            global_step = 0
        if accelerator.is_main_process:
            print(f"从 checkpoint 恢复: {resume_from_checkpoint} (step={global_step})")

    progress_bar = tqdm(
        range(global_step, num_train_steps),
        disable=not accelerator.is_local_main_process,
        initial=global_step,
    )
    best_eval_loss = float("inf")

    # ---------- 5. 训练循环 ----------
    for epoch in range(num_train_epochs):
        policy.train()
        for batch in train_dataloader:
            with accelerator.accumulate(policy):
                pi_chosen = compute_sequence_logps(
                    policy, batch["chosen_input_ids"],
                    batch["chosen_labels"], batch["chosen_attention_mask"],
                )
                pi_rejected = compute_sequence_logps(
                    policy, batch["rejected_input_ids"],
                    batch["rejected_labels"], batch["rejected_attention_mask"],
                )
                with torch.no_grad():
                    ref_chosen = compute_sequence_logps(
                        ref_model, batch["chosen_input_ids"],
                        batch["chosen_labels"], batch["chosen_attention_mask"],
                    )
                    ref_rejected = compute_sequence_logps(
                        ref_model, batch["rejected_input_ids"],
                        batch["rejected_labels"], batch["rejected_attention_mask"],
                    )
                loss, acc = dpo_loss(
                    pi_chosen, pi_rejected, ref_chosen, ref_rejected,
                    beta=beta, label_smoothing=label_smoothing,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                cur_lr = lr_scheduler.get_last_lr()[0]
                progress_bar.set_description(
                    f"Epoch {epoch + 1} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} | Acc: {acc:.3f} | lr: {cur_lr:.2e}"
                )

                # ---- 评估 ----
                if global_step % eval_steps == 0:
                    policy.eval()
                    eval_loss = 0.0
                    eval_acc = 0.0
                    eval_count = 0
                    with torch.no_grad():
                        for eval_batch in eval_dataloader:
                            pi_c = compute_sequence_logps(
                                policy, eval_batch["chosen_input_ids"],
                                eval_batch["chosen_labels"],
                                eval_batch["chosen_attention_mask"],
                            )
                            pi_r = compute_sequence_logps(
                                policy, eval_batch["rejected_input_ids"],
                                eval_batch["rejected_labels"],
                                eval_batch["rejected_attention_mask"],
                            )
                            ref_c = compute_sequence_logps(
                                ref_model, eval_batch["chosen_input_ids"],
                                eval_batch["chosen_labels"],
                                eval_batch["chosen_attention_mask"],
                            )
                            ref_r = compute_sequence_logps(
                                ref_model, eval_batch["rejected_input_ids"],
                                eval_batch["rejected_labels"],
                                eval_batch["rejected_attention_mask"],
                            )
                            el, ea = dpo_loss(pi_c, pi_r, ref_c, ref_r,
                                              beta=beta,
                                              label_smoothing=label_smoothing)
                            loss_gathered = accelerator.gather(el.unsqueeze(0))
                            acc_gathered = accelerator.gather(
                                torch.tensor(ea).unsqueeze(0))
                            eval_loss += loss_gathered.sum().item()
                            eval_acc += acc_gathered.sum().item()
                            eval_count += loss_gathered.size(0)
                    if eval_count == 0:
                        if accelerator.is_main_process:
                            print(f"\nStep {global_step} | 评估集为空，跳过评估")
                        policy.train()
                        continue
                    eval_loss /= eval_count
                    eval_acc /= eval_count
                    if accelerator.is_main_process:
                        print(f"\nStep {global_step} | Eval Loss: {eval_loss:.4f} | "
                              f"Eval Acc: {eval_acc:.3f}")
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process:
                            best_dir = os.path.join(output_dir, "best_dpo")
                            accelerator.unwrap_model(policy).save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            print(f"保存最优 DPO 模型 -> {best_dir}")
                    policy.train()

                # ---- 保存 ----
                if global_step % save_steps == 0:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.wait_for_everyone()
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        accelerator.unwrap_model(policy).save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        print(f"保存 checkpoint -> {ckpt_dir}")

    # ---------- 6. 最终保存 ----------
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final_dpo")
        accelerator.unwrap_model(policy).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        train_args = dict(
            max_seq_len=max_seq_len, batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate, weight_decay=weight_decay,
            warmup_steps=warmup_steps, num_train_epochs=num_train_epochs,
            max_grad_norm=max_grad_norm, beta=beta,
            label_smoothing=label_smoothing, mixed_precision=mixed_precision,
        )
        with open(os.path.join(final_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(train_args, f, ensure_ascii=False, indent=2)
        print(f"DPO 训练完成，最终模型 -> {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MysticMirror DPO 偏好对齐训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---------- 路径 ----------
    parser.add_argument(
        "--base_model_dir", type=str, default="./sft_output/best_sft",
        help="基座模型目录（SFT 后的模型效果更好）",
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/dpo.jsonl",
        help="DPO 数据文件路径（json / jsonl）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dpo_output", help="输出目录",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="从 checkpoint 目录恢复训练（含 optimizer/scheduler 状态），"
             "形如 ./dpo_output/checkpoint-2000",
    )
    # ---------- 训练超参 ----------
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=2, help="每张卡 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="学习率（DPO 通常比 SFT 小 10 倍）")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="warmup 预热步数")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO 温度系数（越大越强调与参考模型的偏差）")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="DPO 标签平滑（0 或 0.1）")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估间隔步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存 checkpoint 间隔步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"], help="混合精度")
    args = parser.parse_args()

    train_dpo(
        base_model_dir=args.base_model_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=args.max_grad_norm,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
