# -*- coding: utf-8 -*-
import os
import math
import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed, GradientAccumulationPlugin
from tqdm.auto import tqdm
from typing import Dict, List, Optional
import warnings

from model_chat import ChatForCausalLM, ChatConfig

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
    model,
    num_train_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
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

class PretrainDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if tokenizer.eos_token_id is None:
            raise ValueError("分词器缺少 eos_token_id，请检查分词器配置！")

        all_tokens: List[int] = []
        for text in tqdm(texts, desc="Tokenizing pretrain data"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)

        self.samples = []
        for i in range(0, len(all_tokens) - max_seq_len + 1, max_seq_len):
            chunk = all_tokens[i: i + max_seq_len]
            self.samples.append(torch.tensor(chunk, dtype=torch.long))

        if len(self.samples) == 0:
            warnings.warn("数据集样本为空！请检查文本长度或 max_seq_len。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = self.samples[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }

def train_pretrain(
    config: ChatConfig,
    data_path: str,
    tokenizer_path: str,
    output_dir: str = "./pretrain_output",
    max_seq_len: int = 2048,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 2000,
    num_train_epochs: int = 1,
    max_grad_norm: float = 1.0,
    eval_steps: int = 500,
    save_steps: int = 2000,
    seed: int = 42,
    mixed_precision: str = "bf16",
    resume_from_checkpoint: Optional[str] = None,
):
    set_seed(seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_dir=output_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")
        print(f"分词器路径: {tokenizer_path}")
        print(f"数据文件: {data_path}")
        print(f"混合精度: {mixed_precision}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if accelerator.is_main_process:
        print(f"分词器加载成功，词表大小: {tokenizer.vocab_size}, "
              f"eos_id: {tokenizer.eos_token_id}")
        if tokenizer.vocab_size != config.vocab_size:
            warnings.warn(
                f"词表不匹配！tokenizer.vocab_size={tokenizer.vocab_size}, "
                f"model.vocab_size={config.vocab_size}"
            )

    model = ChatForCausalLM(config)
    if accelerator.is_main_process:
        print_trainable_parameters(model)

    dataset = load_dataset("text", data_files=data_path, split="train")
    split = dataset.train_test_split(test_size=0.01, seed=seed)
    train_dataset = PretrainDataset(split["train"]["text"], tokenizer, max_seq_len)
    eval_dataset = PretrainDataset(split["test"]["text"], tokenizer, max_seq_len)

    if accelerator.is_main_process:
        print(f"训练样本数: {len(train_dataset)}, 评估样本数: {len(eval_dataset)}")

    num_workers = 0 if os.name == "nt" else 4
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch * num_train_epochs
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model, num_train_steps, learning_rate, weight_decay, warmup_steps
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if resume_from_checkpoint is not None:
        if accelerator.is_main_process:
            print(f"从 checkpoint 恢复: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(os.path.basename(resume_from_checkpoint).split("-")[-1])
    else:
        global_step = 0

    progress_bar = tqdm(
        range(global_step, num_train_steps),
        disable=not accelerator.is_local_main_process,
        initial=global_step,
    )
    best_eval_loss = float("inf")

    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
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
                            loss_gathered = accelerator.gather(outputs.loss)
                            eval_loss += loss_gathered.sum().item()
                            eval_count += loss_gathered.shape[0]
                    eval_loss /= max(eval_count, 1)
                    perplexity = math.exp(min(eval_loss, 20))

                    if accelerator.is_main_process:
                        print(
                            f"\nStep {global_step} | Eval Loss: {eval_loss:.4f} | "
                            f"Perplexity: {perplexity:.2f}"
                        )

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process:
                            best_dir = os.path.join(output_dir, "best_pretrain")
                            accelerator.unwrap_model(model).save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            print(f"保存最优模型 eval_loss={eval_loss:.4f} -> {best_dir}")
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
        final_dir = os.path.join(output_dir, "final_pretrain")
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        train_args = dict(
            max_seq_len=max_seq_len, batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate, weight_decay=weight_decay,
            warmup_steps=warmup_steps, num_train_epochs=num_train_epochs,
            max_grad_norm=max_grad_norm, mixed_precision=mixed_precision,
        )
        with open(os.path.join(final_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(train_args, f, ensure_ascii=False, indent=2)
        print(f"预训练完成，最终模型已保存到 {final_dir}")


if __name__ == "__main__":
    model_config = ChatConfig(
        vocab_size=6400,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    )

    train_pretrain(
        config=model_config,
        data_path="./data/train_data.txt",
        tokenizer_path="./tokenizer/",
        output_dir="./pretrain_output",
        max_seq_len=2048,
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=100,
        num_train_epochs=1,
        max_grad_norm=1.0,
        eval_steps=500,
        save_steps=100,
        seed=42,
        mixed_precision="bf16",
        resume_from_checkpoint=None,
    )