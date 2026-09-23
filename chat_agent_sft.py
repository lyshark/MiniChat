# -*- coding: utf-8 -*-
"""
MysticMirror Agent SFT 对话测试脚本（命令行参数启动）
====================================================

用法示例：
    python chat_agent_sft.py
    python chat_agent_sft.py --model_path ./agent_sft_output/best_agent_sft
    python chat_agent_sft.py --model_path ./sft_output/best_sft \
        --device cuda --dtype bf16 --max_new_tokens 512
    python chat_agent_sft.py --greedy --no_history
"""
import argparse
import ast
import datetime
import json
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

# ============================================================
# Agent 对话模板
# ============================================================
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
OBS_START = "<|observation|>"
OBS_END = "<|observation_end|>"
GEN_PROMPT = f"{IM_START}assistant\n"

_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END),
    re.DOTALL,
)

DEFAULT_SYSTEM_PROMPT = (
    "你是一个可以调用工具的智能助手。你可以使用以下工具：\n"
    "- get_system_time：获取当前系统时间，参数 {}；\n"
    "- get_system_date：获取当前日期，参数 {}；\n"
    "- get_system_datetime：获取当前日期和时间，参数 {}；\n"
    "- get_weather：查询城市天气，参数 {\"city\": \"城市名\"}；\n"
    "- calculator：数学计算，参数 {\"expr\": \"算式\"}，例如 {\"expr\": \"20 * 16 - 45\"}。\n"
    "当用户询问实时信息（时间、日期、天气、计算等）时，请先调用工具获取真实结果，"
    "不要自己编造信息。工具调用格式："
    "<|tool_call_start|>工具名{\"参数\":\"值\"}<|tool_call_end|>。"
)

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


# ============================================================
# 工具注册表
# ============================================================
def _tool_get_system_time(args):
    return datetime.datetime.now().strftime("%H:%M:%S")


def _tool_get_system_date(args):
    return datetime.datetime.now().strftime("%Y-%m-%d")


def _tool_get_system_datetime(args):
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_WEATHER_TABLE = {
    "北京": "晴，22~30℃", "上海": "多云，22~28℃", "广州": "雷阵雨，27~32℃",
    "深圳": "小雨，26~31℃", "杭州": "晴，21~29℃", "南京": "阴，20~27℃",
    "武汉": "多云，23~30℃", "成都": "阴天，19~25℃", "西安": "晴，18~29℃",
    "长沙": "小雨，20~26℃", "厦门": "多云，25~31℃", "泰安": "晴，19~27℃",
}


def _tool_get_weather(args):
    city = str(args.get("city", "")).strip() or "未知城市"
    return _WEATHER_TABLE.get(city, f"{city}多云，22~28℃")


def _safe_calc(expr: str):
    """仅允许数字与四则/幂运算的安全计算器（基于 AST 白名单）。"""
    tree = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("表达式包含不允许的运算")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("只允许数字常量")
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}})


def _tool_calculator(args):
    expr = str(args.get("expr", "")).strip()
    if not expr:
        raise ValueError("缺少 expr 参数")
    result = _safe_calc(expr)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)

TOOL_REGISTRY = {
    "get_system_time": (_tool_get_system_time, "获取当前系统时间"),
    "get_system_date": (_tool_get_system_date, "获取当前日期"),
    "get_system_datetime": (_tool_get_system_datetime, "获取当前日期和时间"),
    "get_weather": (_tool_get_weather, "查询天气，参数 {\"city\":\"城市\"}"),
    "calculator": (_tool_calculator, "数学计算，参数 {\"expr\":\"算式\"}"),
}

# ============================================================
# 参数与运行环境
# ============================================================
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
    v = float(value)
    if not (0.0 < v <= 1.0):
        raise argparse.ArgumentTypeError(f"必须落在 (0, 1]，得到 {v}")
    return v


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MysticMirror Agent SFT 对话测试脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---------- 模型与运行环境 ----------
    parser.add_argument(
        "--model_path", type=str, default="./agent_sft_output/best_agent_sft",
        help="Agent SFT 模型目录（须包含 config.json、model.safetensors 及 tokenizer 文件）",
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
        "--max_new_tokens", type=_positive_int, default=256,
        help="每轮生成的最大 token 数（>0）",
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
        "--repetition_penalty", type=_positive_float, default=1.0,
        help="重复惩罚系数（>0）",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="贪心解码（关闭随机采样，等效 temperature=0）",
    )
    parser.add_argument(
        "--max_tool_rounds", type=_positive_int, default=8,
        help="单次用户提问最多允许的工具调用轮数",
    )
    parser.add_argument(
        "--no_history", action="store_true",
        help="关闭多轮历史，每轮独立单轮对话",
    )
    parser.add_argument(
        "--max_prompt_len", type=_positive_int, default=None,
        help="输入上下文最大 token 数（历史 + 当前输入）；默认自动取 "
             "max_position_embeddings - max_new_tokens，避免超长触发 RoPE 外推警告",
    )
    # ---------- Agent 参数 ----------
    parser.add_argument(
        "--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
        help="系统提示词；传空字符串 '' 表示不使用 system 消息",
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
        raise FileNotFoundError(
            f"模型目录不存在: {model_path}\n"
            "请先完成 Agent SFT 训练（生成 ./agent_sft_output/best_agent_sft），"
            "或通过 --model_path 指定已有的模型目录。"
        )

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


# ============================================================
# Agent 对话引擎
# ============================================================
class AgentChat:
    """Agent 多轮对话引擎（含工具调用循环）。

    - 消息渲染与 train_agent_sft.py 的模板逐字一致；
    - system 消息固定保留，超长时先丢弃最早的非 system 消息，
      仍超长再按 token 保留尾部；
    - 提示 + 生成长度不超过 max_position_embeddings，避免 RoPE 外推告警。
    """

    _HISTORY_HEADROOM = 64

    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.messages = []
        if args.system_prompt:
            self.messages.append({"role": "system", "content": args.system_prompt})

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

    # ---------- 模板渲染 ----------
    def _render_messages(self, messages):
        """把消息列表渲染成训练模板文本（不含生成提示后缀）。"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "") or ""
            if role == "system":
                parts.append(f"{IM_START}system\n{content}{IM_END}\n")
            elif role == "user":
                parts.append(f"{IM_START}user\n{content}{IM_END}\n")
            elif role == "tool":
                obs_name = msg.get("name", "")
                parts.append(
                    f"{IM_START}tool\n[{obs_name}]{OBS_START}{content}{OBS_END}{IM_END}\n"
                )
            elif role == "assistant":
                parts.append(f"{IM_START}assistant\n{content}")
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    parts.append(
                        f"{TOOL_CALL_START}{fn.get('name', '')}"
                        f"{fn.get('arguments', '{}')}{TOOL_CALL_END}"
                    )
                parts.append(f"{IM_END}\n")
        return "".join(parts)

    def _render_prompt(self):
        return self._render_messages(self.messages) + GEN_PROMPT

    # ---------- 历史窗口裁剪 ----------
    def _prune_messages(self):
        """历史消息自身不超出预算（为当前输入预留头部空间）。"""
        budget = max(self.max_prompt_len - self._HISTORY_HEADROOM, 0)
        while True:
            hist_ids = self.tokenizer.encode(
                self._render_messages(self.messages), add_special_tokens=False
            )
            if len(hist_ids) <= budget:
                return
            for i, msg in enumerate(self.messages):
                if msg["role"] != "system":
                    self.messages.pop(i)
                    break
            else:
                return

    def _build_prompt_ids(self):
        for _ in range(len(self.messages) + 32):
            prompt = self._render_prompt()
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) <= self.max_prompt_len:
                return ids
            for i, msg in enumerate(self.messages):
                if msg["role"] != "system":
                    self.messages.pop(i)
                    break
            else:
                return ids[-self.max_prompt_len:]
        return self.tokenizer.encode(self._render_prompt(),
                                     add_special_tokens=False)[-self.max_prompt_len:]

    # ---------- 工具调用 ----------
    def _split_call(self, body):
        body = body.strip()
        for name in sorted(TOOL_REGISTRY, key=len, reverse=True):
            if body.startswith(name):
                return name, body[len(name):].strip()
        idx = body.find("{")
        if idx > 0:
            return body[:idx].strip(), body[idx:].strip()
        return body, "{}"

    def _parse_tool_calls(self, text):
        """从生成文本中解析出 [(工具名, 参数字典), ...]。"""
        calls = []
        for m in _TOOL_CALL_RE.finditer(text):
            name, args_text = self._split_call(m.group(1))
            try:
                args = json.loads(args_text) if args_text else {}
                if not isinstance(args, dict):
                    args = {"value": args}
            except json.JSONDecodeError:
                args = {"raw": args_text}
            calls.append((name, args))
        return calls

    def _call_tool(self, name, args):
        if name not in TOOL_REGISTRY:
            raise ValueError(f"未知工具: {name}，可用工具: {sorted(TOOL_REGISTRY)}")
        func, desc = TOOL_REGISTRY[name]
        result = func(args)
        if not isinstance(result, str):
            result = str(result)
        return result

    # ---------- 单轮生成 ----------
    def _generate_once(self):
        prompt_ids = self._build_prompt_ids()
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
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        text = _BUFFER_RE.sub("", text).strip()
        return text

    # ---------- 主入口 ----------
    def chat(self, user_input, verbose=True):
        """一轮用户提问：内部循环调用工具直到给出最终回答。"""
        user_input = user_input.strip()
        if not user_input:
            raise ValueError("输入不能为空！")

        self.messages.append({"role": "user", "content": user_input})
        final_answer = ""

        for _ in range(self.args.max_tool_rounds):
            text = self._generate_once()
            tool_calls = self._parse_tool_calls(text)
            content = _TOOL_CALL_RE.sub("", text).strip()

            if not tool_calls:
                final_answer = text
                self.messages.append({"role": "assistant", "content": text})
                break

            self.messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"function": {"name": name, "arguments": json.dumps(
                        args, ensure_ascii=False)}}
                    for name, args in tool_calls
                ],
            })

            for name, args in tool_calls:
                try:
                    result = self._call_tool(name, args)
                    if verbose:
                        print(f"  [工具调用] {name} {json.dumps(args, ensure_ascii=False)}"
                              f" -> {result}")
                except Exception as exc:
                    result = f"工具执行失败：{exc}"
                    if verbose:
                        print(f"  [工具调用失败] {name} {json.dumps(args, ensure_ascii=False)}"
                              f" -> {result}")
                self.messages.append({"role": "tool", "name": name, "content": result})
        else:
            final_answer = final_answer or "(达到最大工具调用轮数，已停止)"
            if verbose:
                print("  [提示] 达到最大工具调用轮数")

        if not self.args.no_history:
            self._prune_messages()
        else:
            self.messages = [m for m in self.messages if m["role"] == "system"]
        return final_answer

    def __call__(self, user_input, verbose=True):
        return self.chat(user_input, verbose=verbose)

def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args)
    dtype = resolve_dtype(args, device)
    model, tokenizer = load_model(args.model_path, device, dtype)
    engine = AgentChat(model, tokenizer, device, args)

    history_note = "已开启多轮历史" if not args.no_history else "单轮模式（无历史）"
    print(f"\nAgent 对话测试开始（{history_note}，工具："
          f"{', '.join(sorted(TOOL_REGISTRY))}）。"
          "输入 exit / quit 退出。\n")
    while True:
        try:
            user_input = input("User：")
        except (EOFError, KeyboardInterrupt):
            print("\n输入结束，退出。")
            break
        if user_input.strip().lower() in ("exit", "quit"):
            print("退出对话")
            break
        if not user_input.strip():
            continue
        try:
            response = engine(user_input)
        except Exception as exc:
            print(f"[错误] 本轮生成失败：{exc}\n")
            continue
        print(f"Assistant：{response}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())