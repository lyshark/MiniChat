# -*- coding: utf-8 -*-
import argparse
import os
import math
import json
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from typing import Dict, List, Optional
import warnings
from modeling_mystic_mirror import (
    MysticMirrorConfig,
    MysticMirrorForCausalLM,
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_pos_emb,
    repeat_kv,
    build_causal_allowed,
    unpack_past_cache,
    validate_input_ids,
    validate_attention_mask,
    validate_sampling_params,
    normalize_eos_token_id,
    apply_repetition_penalty,
    warp_logits,
    sample_next_token,
)

from transformers import AutoConfig, AutoModelForCausalLM
from modeling_mystic_mirror import MysticMirrorConfig, MysticMirrorForCausalLM

# 注册模型 消除警告
AutoConfig.register("MysticMirror", MysticMirrorConfig)
AutoModelForCausalLM.register(MysticMirrorConfig, MysticMirrorForCausalLM)

# ============================================================
# 工具函数
# ============================================================
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"可训练参数: {trainable_params:,} || 总参数: {all_param:,} || "
        f"占比: {100 * trainable_params / all_param:.2f}%"
    )


def create_optimizer_and_scheduler(
    model, num_train_steps, learning_rate, weight_decay, warmup_steps
):
    no_decay = ["bias", "RMSNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                      betas=(0.9, 0.95), eps=1e-8)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_steps,
    )
    return optimizer, lr_scheduler


# ============================================================
# SFT 数据集
# ============================================================
class SFTDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_seq_len: int,
                 template_name: str = "MysticMirror"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.template_name = template_name
        if tokenizer.eos_token_id is None:
            raise ValueError("分词器缺少 eos_token_id！")
        self.samples: List[Dict[str, torch.Tensor]] = []
        self._build_samples(data_list)

    def _build_samples(self, data_list):
        for item in tqdm(data_list, desc="Building SFT samples"):
            sample = self._format_example(item)
            if sample is not None:
                self.samples.append(sample)
        if len(self.samples) == 0:
            warnings.warn("SFT 数据集为空！请检查数据格式。")

    # ---- 把单条原始数据转成 messages 列表 ----
    def _to_messages(self, item) -> List[Dict[str, str]]:
        if "messages" in item:
            # OpenAI / ShareGPT 格式
            return item["messages"]
        elif "instruction" in item or "output" in item:
            # Alpaca 格式
            instruction = item.get("instruction", "")
            user_input = item.get("input", "")
            output = item.get("output", "")
            user_content = instruction
            if user_input:
                user_content = f"{instruction}\n{user_input}"
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
        else:
            warnings.warn(f"无法识别的数据格式，跳过: {list(item.keys())}")
            return []

    # ---- 用 MysticMirror template 格式化并标记损失掩码 ----
    def _format_example(self, item) -> Optional[Dict[str, torch.Tensor]]:
        messages = self._to_messages(item)
        if not messages:
            return None
        tokenizer = self.tokenizer
        input_ids_list: List[int] = []
        labels_list: List[int] = []
        for turn_idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            if hasattr(tokenizer, "apply_MysticMirror_template"):
                prefix_msgs = messages[: turn_idx + 1]
                rendered = tokenizer.apply_MysticMirror_template(
                    prefix_msgs, tokenize=False, add_generation_prompt=False
                )
                prefix_only = tokenizer.apply_MysticMirror_template(
                    messages[:turn_idx], tokenize=False, add_generation_prompt=(role == "assistant")
                )
                full_so_far = tokenizer.apply_MysticMirror_template(
                    prefix_msgs, tokenize=False, add_generation_prompt=False
                )
                prefix_ids = tokenizer.encode(prefix_only, add_special_tokens=False)
                full_ids = tokenizer.encode(full_so_far, add_special_tokens=False)
                current_ids = full_ids[len(prefix_ids):]
            else:
                if role == "user":
                    prefix_text = f"<|im_start|>user\n{content}<|im_end|>\n"
                    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                    current_ids = []
                else:  # assistant
                    prefix_text = f"<|im_start|>assistant\n"
                    content_text = f"{content}<|im_end|>\n"
                    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                    current_ids = tokenizer.encode(content_text, add_special_tokens=False)

            input_ids_list.extend(prefix_ids)
            labels_list.extend([-100] * len(prefix_ids))
            if role == "assistant":
                input_ids_list.extend(current_ids)
                labels_list.extend(current_ids)
        if len(input_ids_list) > self.max_seq_len:
            input_ids_list = input_ids_list[: self.max_seq_len]
            labels_list = labels_list[: self.max_seq_len]
        if all(l == -100 for l in labels_list):
            return None
        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids_list), dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id: int):
    """动态 padding collate：把 batch 内的样本 pad 到等长。"""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        seq_len = len(x["input_ids"])
        pad_len = max_len - seq_len
        input_ids.append(torch.cat([x["input_ids"],
                                    torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([x["labels"],
                                 torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([x["attention_mask"],
                                         torch.zeros(pad_len, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


# ============================================================
# SFT 训练主函数
# ============================================================
def train_sft(
    pretrain_model_dir: str,
    data_path: str,
    output_dir: str = "./sft_output",
    max_seq_len: int = 2048,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    warmup_steps: int = 100,
    num_train_epochs: int = 3,
    max_grad_norm: float = 1.0,
    eval_steps: int = 100,
    save_steps: int = 500,
    seed: int = 42,
    mixed_precision: str = "bf16",
):
    """MysticMirror SFT 训练主入口。
    Args:
        pretrain_model_dir: 预训练模型目录（绝对路径，含 config.json 和 model weights）。
        data_path: SFT 数据文件（json / jsonl）。
        output_dir: 输出目录。
        max_seq_len: 最大序列长度。
        batch_size: 每张卡 batch size。
        gradient_accumulation_steps: 梯度累积。
        learning_rate: 学习率（SFT 通常比预训练小 10x）。
        weight_decay: weight decay。
        warmup_steps: 预热步数。
        num_train_epochs: 训练轮数。
        max_grad_norm: 梯度裁剪。
        eval_steps: 评估间隔。
        save_steps: 保存间隔。
        seed: 随机种子。
        mixed_precision: 混合精度。
    """
    set_seed(seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_dir=output_dir,
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"预训练模型: {pretrain_model_dir}")
        print(f"SFT 数据: {data_path}")
        print(f"输出目录: {output_dir}")

    # ---------- 加载分词器和模型 ----------
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dir)
    model = MysticMirrorForCausalLM.from_pretrained(
        pretrain_model_dir,
        torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float32,
    )
    if accelerator.is_main_process:
        print_trainable_parameters(model)

    # ---------- 数据 ----------
    raw_data = load_dataset("json", data_files=data_path, split="train")
    split = raw_data.train_test_split(test_size=0.05, seed=seed)
    train_dataset = SFTDataset(split["train"], tokenizer, max_seq_len)
    eval_dataset = SFTDataset(split["test"], tokenizer, max_seq_len)
    if accelerator.is_main_process:
        print(f"训练样本: {len(train_dataset)}, 评估样本: {len(eval_dataset)}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    num_workers = 0 if os.name == "nt" else 4
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # ---------- 优化器 ----------
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch * num_train_epochs
    warmup = min(warmup_steps, int(num_train_steps * 0.05))
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model, num_train_steps, learning_rate, weight_decay, warmup
    )

    # ---------- prepare ----------
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(num_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    best_eval_loss = float("inf")

    # ---------- 训练循环 ----------
    for epoch in range(num_train_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                cur_lr = lr_scheduler.get_last_lr()[0]
                progress_bar.set_description(
                    f"Epoch {epoch + 1} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} | lr: {cur_lr:.2e}"
                )

                # ---- 评估 ----
                if global_step % eval_steps == 0:
                    model.eval()
                    eval_loss = 0.0
                    eval_count = 0
                    with torch.no_grad():
                        for eval_batch in eval_dataloader:
                            outputs = model(**eval_batch)
                            loss_gathered = accelerator.gather(outputs.loss.unsqueeze(0))
                            eval_loss += loss_gathered.sum().item()
                            eval_count += loss_gathered.size(0)
                    eval_loss /= max(eval_count, 1)
                    if accelerator.is_main_process:
                        print(f"\nStep {global_step} | Eval Loss: {eval_loss:.4f}")
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process:
                            best_dir = os.path.join(output_dir, "best_sft")
                            accelerator.unwrap_model(model).save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            print(f"保存最优 SFT 模型 -> {best_dir}")
                    model.train()

                # ---- 保存 ----
                if global_step % save_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"保存 checkpoint -> {ckpt_dir}")

    # ---------- 最终保存 ----------
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final_sft")
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"SFT 训练完成，最终模型 -> {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MysticMirror SFT 微调脚本，读取 json/jsonl 对话数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---------- 路径 ----------
    parser.add_argument(
        "--pretrain_model_dir", type=str, default="./pretrain_output/best_pretrain",
        help="预训练模型目录（含 config.json 和权重）",
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/sft_alpaca.jsonl",
        help="SFT 数据文件路径（json / jsonl，OpenAI messages 或 Alpaca 格式）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./sft_output", help="输出目录",
    )
    # ---------- 训练超参 ----------
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="每张卡 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率（SFT 通常比预训练小 10 倍）")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="warmup 预热步数")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估间隔步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存 checkpoint 间隔步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"], help="混合精度")
    args = parser.parse_args()

    train_sft(
        pretrain_model_dir=args.pretrain_model_dir,
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
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
    )
