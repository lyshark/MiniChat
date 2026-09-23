# -*- coding: utf-8 -*-
"""
用法示例：
    python chat_pretrain.py
    python chat_pretrain.py --model_path ./pretrain_output/best_pretrain
    python chat_pretrain.py --model_path ./pretrain_output/final_pretrain \
        --device cuda --dtype bf16 --max_new_tokens 512 --temperature 0.8
    python chat_pretrain.py --greedy --max_new_tokens 64
"""
import argparse
import os
import re
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

from modeling_mystic_mirror import (
    MysticMirrorConfig,
    MysticMirrorForCausalLM,
)

AutoConfig.register("MysticMirror", MysticMirrorConfig)

_BUFFER_RE = re.compile(r"<\|buffer\d+\|>")

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _positive_int(value):
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"必须 > 0，得到 {v}")
    return v


def _nonneg_int(value):
    v = int(value)
    if v < 0:
        raise argparse.ArgumentTypeError(f"必须 >= 0，得到 {v}")
    return v


def _positive_float(value):
    v = float(value)
    if not (v > 0):
        raise argparse.ArgumentTypeError(f"必须 > 0，得到 {v}")
    return v


def _nonneg_float(value):
    v = float(value)
    if v < 0:
        raise argparse.ArgumentTypeError(f"必须 >= 0，得到 {v}")
    return v


def _prob_float(value):
    """核采样阈值，取值 (0, 1]。"""
    v = float(value)
    if not (0.0 < v <= 1.0):
        raise argparse.ArgumentTypeError(f"必须落在 (0, 1]，得到 {v}")
    return v


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MysticMirror 预训练模型续写 / 对话脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---------- 模型与运行环境 ----------
    parser.add_argument(
        "--model_path", type=str, default="./pretrain_output/best_pretrain",
        help="模型目录（须包含 config.json、model.safetensors 及 tokenizer 文件）",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="推理设备；auto 表示有 CUDA 用 CUDA，否则用 CPU",
    )
    parser.add_argument(
        "--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"],
        help="推理精度；auto 在 CUDA 上用 bf16、在 CPU 上用 fp32",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="随机种子，用于复现采样结果",
    )
    # ---------- 生成参数 ----------
    parser.add_argument(
        "--max_new_tokens", type=_positive_int, default=256, help="最多生成的新 token 数（>0）",
    )
    parser.add_argument(
        "--temperature", type=_nonneg_float, default=0.7, help="采样温度（>=0）",
    )
    parser.add_argument(
        "--top_p", type=_prob_float, default=0.9, help="核采样阈值（0, 1]",
    )
    parser.add_argument(
        "--top_k", type=_nonneg_int, default=50, help="top-k 采样个数（0 表示关闭）",
    )
    parser.add_argument(
        "--repetition_penalty", type=_positive_float, default=1.1, help="重复惩罚系数（>0）",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="贪心解码（关闭随机采样，等效 temperature=0）",
    )
    parser.add_argument(
        "--max_prompt_len", type=_positive_int, default=None,
        help="输入上下文最大 token 数（历史 + 当前输入）；默认自动取 "
             "max_position_embeddings - max_new_tokens，避免超长触发 RoPE 外推警告",
    )
    return parser.parse_args(argv)


def resolve_device(args):
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[提示] 未检测到可用 CUDA，已回退到 CPU。")
        return "cpu"
    return args.device


def resolve_dtype(args, device):
    if args.dtype != "auto":
        dtype = _DTYPE_MAP[args.dtype]
        if device == "cpu" and dtype != torch.float32:
            print(f"[提示] CPU 上使用 {args.dtype} 可能较慢或部分算子不受支持，"
                  "建议改用 --dtype fp32。")
        return dtype
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model(model_path, device, dtype):
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    print(f"加载分词器：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("分词器缺少 eos_token_id，请检查 tokenizer 配置！")

    print(f"加载模型配置：{model_path}")
    lm_config = MysticMirrorConfig.from_pretrained(model_path)

    print(f"加载模型权重（device={device}, dtype={dtype}）：{model_path}")
    model = MysticMirrorForCausalLM.from_pretrained(
        model_path,
        config=lm_config,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"模型加载完成：参数量 {num_params:,} | vocab={lm_config.vocab_size} | "
        f"layers={lm_config.num_hidden_layers} | "
        f"max_position_embeddings={lm_config.max_position_embeddings}"
    )
    return model, tokenizer

class MysticMirror:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.history_ids = []

        max_pos = model.config.max_position_embeddings
        if args.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens 必须 > 0，得到 {args.max_new_tokens}")
        if args.max_new_tokens >= max_pos:
            raise ValueError(
                f"max_new_tokens({args.max_new_tokens}) 必须小于 "
                f"max_position_embeddings({max_pos})"
            )
        auto_budget = max_pos - args.max_new_tokens
        self.max_prompt_len = args.max_prompt_len or auto_budget
        if self.max_prompt_len <= 0:
            raise ValueError(f"max_prompt_len 必须 > 0，得到 {self.max_prompt_len}")
        self.max_prompt_len = min(self.max_prompt_len, auto_budget)

    def generate(self, user_input):
        inp_ids = self.tokenizer.encode(user_input, add_special_tokens=False)

        if len(inp_ids) > self.max_prompt_len:
            inp_ids = inp_ids[-self.max_prompt_len:]

        budget = self.max_prompt_len - len(inp_ids)
        if budget <= 0:
            self.history_ids = []
        elif len(self.history_ids) > budget:
            self.history_ids = self.history_ids[-budget:]

        prompt_ids = self.history_ids + inp_ids
        prompt_tensor = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.device
        )
        input_len = prompt_tensor.shape[-1]
        outputs = self.model.generate(
            input_ids=prompt_tensor,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            do_sample=not self.args.greedy,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.args.repetition_penalty,
        )
        new_ids = outputs[0][input_len:].tolist()

        response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        response = _BUFFER_RE.sub("", response).strip()
        self.history_ids = (prompt_ids + new_ids)[-self.max_prompt_len:]
        return response

    def __call__(self, user_input):
        return self.generate(user_input)

def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args)
    dtype = resolve_dtype(args, device)
    model, tokenizer = load_model(args.model_path, device, dtype)
    engine = MysticMirror(model, tokenizer, device, args)

    print("\n输入 exit / quit 退出；直接回车表示基于已有历史继续续写。\n")
    while True:
        try:
            user_input = input("请输入文本：")
        except (EOFError, KeyboardInterrupt):
            print("\n输入结束，退出。")
            break
        if user_input.strip().lower() in ("exit", "quit"):
            break
        if not user_input.strip() and not engine.history_ids:
            print("[提示] 历史为空且输入为空，请先输入一些文本。")
            continue
        try:
            response = engine(user_input)
        except Exception as exc:
            print(f"[错误] 本轮生成失败：{exc}")
            continue
        print(f"续写结果：{response}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())