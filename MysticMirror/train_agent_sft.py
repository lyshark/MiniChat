# -*- coding: utf-8 -*-
import argparse
import json
import os
import math
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_scheduler, AutoConfig, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from typing import Dict, List, Optional
import warnings

from modeling_mystic_mirror import (
    MysticMirrorConfig,
    MysticMirrorForCausalLM,
)
from train_sft import (
    print_trainable_parameters,
    create_optimizer_and_scheduler,
    collate_fn,
    load_json_or_jsonl,
    split_data,
)

AutoConfig.register("MysticMirror", MysticMirrorConfig)
AutoModelForCausalLM.register(MysticMirrorConfig, MysticMirrorForCausalLM)


# ============================================================
# Agent 对话模板
# ============================================================

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
OBS_START = "<tool_response>"
OBS_END = "</tool_response>"

class AgentSFTDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if tokenizer.eos_token_id is None:
            raise ValueError("分词器缺少 eos_token_id！")

        required = {
            "<|im_start|>": self._get_special_id("<|im_start|>"),
            "<|im_end|>": self._get_special_id("<|im_end|>"),
            "<tool_call>": self._get_special_id("<tool_call>"),
            "</tool_call>": self._get_special_id("</tool_call>"),
            "<tool_response>": self._get_special_id("<tool_response>"),
            "</tool_response>": self._get_special_id("</tool_response>"),
        }
        missing = [tok for tok, tid in required.items() if tid is None]
        if missing:
            raise ValueError(
                f"分词器词表缺少模板标记: {missing}；请先更新 tokenizer "
                "(tokenizer.json / tokenizer_config.json) 后再训练。"
            )
        self.im_start_id = required["<|im_start|>"]
        self.im_end_id = required["<|im_end|>"]
        self.tc_start_id = required["<tool_call>"]
        self.tc_end_id = required["</tool_call>"]
        self.obs_start_id = required["<tool_response>"]
        self.obs_end_id = required["</tool_response>"]

        self.samples: List[Dict[str, torch.Tensor]] = []
        self._build(data_list)

    def _get_special_id(self, token: str) -> Optional[int]:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != self.tokenizer.unk_token_id:
            return tid
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _build(self, data_list):
        for item in tqdm(data_list, desc="Building Agent SFT samples"):
            sample = self._format(item)
            if sample is not None:
                self.samples.append(sample)
        if not self.samples:
            warnings.warn("Agent SFT 数据集为空！")

    # ---- 单条样本格式化 ----
    def _format(self, item) -> Optional[Dict[str, torch.Tensor]]:
        messages = item.get("messages", [])
        if not messages:
            return None

        tok = self.tokenizer
        input_ids: List[int] = []
        labels: List[int] = []

        for msg in messages:
            role = msg.get("role")
            if not role:
                continue
            content = msg.get("content", "") or ""

            if role == "system":
                text = f"{IM_START}system\n{content}{IM_END}\n"
                ids = tok.encode(text, add_special_tokens=False)
                input_ids.extend(ids)
                labels.extend([-100] * len(ids))

            elif role == "user":
                text = f"{IM_START}user\n{content}{IM_END}\n"
                ids = tok.encode(text, add_special_tokens=False)
                input_ids.extend(ids)
                labels.extend([-100] * len(ids))

            elif role == "tool":
                text = f"{IM_START}user\n{OBS_START}\n{content}\n{OBS_END}{IM_END}\n"
                ids = tok.encode(text, add_special_tokens=False)
                input_ids.extend(ids)
                labels.extend([-100] * len(ids))

            elif role == "assistant":
                prefix = f"{IM_START}assistant\n"
                prefix_ids = tok.encode(prefix, add_special_tokens=False)
                input_ids.extend(prefix_ids)
                labels.extend([-100] * len(prefix_ids))

                content_ids = tok.encode(content, add_special_tokens=False) if content else []
                input_ids.extend(content_ids)
                labels.extend(content_ids)

                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "")
                    args = fn.get("arguments", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    call_text = (
                        f"{TOOL_CALL_START}\n"
                        f'{{"name": "{name}", "arguments": {args}}}\n'
                        f"{TOOL_CALL_END}"
                    )
                    tc_ids = tok.encode(call_text, add_special_tokens=False)
                    input_ids.extend(tc_ids)
                    labels.extend(tc_ids)

                end_text = f"{IM_END}\n"
                end_ids = tok.encode(end_text, add_special_tokens=False)
                input_ids.extend(end_ids)
                labels.extend(end_ids)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        if all(l == -100 for l in labels):
            return None

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Agent SFT 训练主函数
# ============================================================

def train_agent_sft(
    base_model_dir: str,
    data_path: str,
    output_dir: str = "./agent_sft_output",
    max_seq_len: int = 2048,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    warmup_steps: int = 100,
    num_train_epochs: int = 3,
    max_grad_norm: float = 1.0,
    eval_steps: int = 100,
    save_steps: int = 500,
    seed: int = 42,
    mixed_precision: str = "bf16",
    resume_from_checkpoint: Optional[str] = None,
):
    """MysticMirror Agent SFT 训练主入口。

    Args:
        base_model_dir: 基座模型（SFT 后的模型效果更好）。
        data_path: Agent 数据（jsonl，含 messages 和 tool_calls）。
        output_dir: 输出目录。
        max_seq_len: 最大序列长度（多轮工具调用需要更长）。
        batch_size: 每张卡 batch size。
        gradient_accumulation_steps: 梯度累积。
        learning_rate: 学习率。
        weight_decay: weight decay。
        warmup_steps: 预热步数。
        num_train_epochs: 训练轮数。
        max_grad_norm: 梯度裁剪。
        eval_steps: 评估间隔。
        save_steps: 保存间隔。
        seed: 随机种子。
        mixed_precision: 混合精度。
        resume_from_checkpoint: 恢复训练的 checkpoint 目录（含 optimizer/scheduler
            状态），目录名形如 checkpoint-2000；不填则从头训练。
    """
    if not os.path.isdir(base_model_dir):
        raise FileNotFoundError(
            f"基座模型目录不存在: {base_model_dir}\n"
            "请先完成 SFT 训练（生成 ./sft_output/best_sft），"
            "或通过 --base_model_dir 指定已有的模型目录。"
        )
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Agent 数据文件不存在: {data_path}")
    if resume_from_checkpoint is not None and not os.path.isdir(resume_from_checkpoint):
        raise FileNotFoundError(f"恢复目录不存在: {resume_from_checkpoint}")

    set_seed(seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_dir=output_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"基座模型: {base_model_dir}")
        print(f"Agent SFT 数据: {data_path}")
        print(f"输出目录: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    model = MysticMirrorForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float32,
    )

    if accelerator.is_main_process:
        print_trainable_parameters(model)

    # ---------- 数据 ----------
    data_list = load_json_or_jsonl(data_path)
    if not data_list:
        raise ValueError(f"数据文件为空或格式无法解析: {data_path}")
    train_data, eval_data = split_data(data_list, eval_ratio=0.05, seed=seed)
    train_dataset = AgentSFTDataset(train_data, tokenizer, max_seq_len)
    eval_dataset = AgentSFTDataset(eval_data, tokenizer, max_seq_len)
    if len(train_dataset) == 0:
        raise ValueError(
            f"训练样本为空（原始 {len(data_list)} 条，0 条可用）！\n"
            "请检查数据格式：须为 OpenAI function calling 风格的 messages "
            "（含 tool_calls 字段）；并确认 max_seq_len 未把全部样本截断。"
        )

    if accelerator.is_main_process:
        print(f"原始数据: {len(data_list)} 条 | 训练样本: {len(train_dataset)}, "
              f"评估样本: {len(eval_dataset)}")

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

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
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

    # ---------- 训练 ----------
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
                    if eval_count == 0:
                        if accelerator.is_main_process:
                            print(f"\nStep {global_step} | 评估集为空，跳过评估")
                        model.train()
                        continue
                    eval_loss /= eval_count

                    if accelerator.is_main_process:
                        print(f"\nStep {global_step} | Eval Loss: {eval_loss:.4f}")

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process:
                            best_dir = os.path.join(output_dir, "best_agent_sft")
                            accelerator.unwrap_model(model).save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            print(f"保存最优 Agent SFT 模型 -> {best_dir}")
                    model.train()

                if global_step % save_steps == 0:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.wait_for_everyone()
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        print(f"保存 checkpoint -> {ckpt_dir}")

    accelerator.wait_for_everyone()
    accelerator.end_training()
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final_agent_sft")
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Agent SFT 训练完成，模型 -> {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MysticMirror Agent SFT 微调脚本（含 tool_calls 的对话数据）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---------- 路径 ----------
    parser.add_argument(
        "--base_model_dir", type=str, default="./sft_output/best_sft",
        help="基座模型目录（SFT 后的模型效果更好）",
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/agent_sft_alpaca.jsonl",
        help="Agent 数据文件路径（json / jsonl，含 messages 与 tool_calls）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./agent_sft_output", help="输出目录",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="从 checkpoint 目录恢复训练（含 optimizer/scheduler 状态），"
             "形如 ./agent_sft_output/checkpoint-2000",
    )
    # ---------- 训练超参 ----------
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=2, help="每张卡 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
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

    train_agent_sft(
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
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
