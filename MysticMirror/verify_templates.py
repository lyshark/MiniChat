# -*- coding: utf-8 -*-
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# ---------- 1. 源码断言 ----------
def read(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as f:
        return f.read()


print("== 1. 源码模板一致性 ==")
sft_src = read("train_sft.py")
check("train_sft.py 已删除 apply_MysticMirror_template 双分支",
      "apply_MysticMirror_template" not in sft_src)
check("train_sft.py 不再依赖 datasets.load_dataset", "load_dataset" not in sft_src)
check("train_sft.py 支持 MiniMind conversations 格式",
      '"conversations" in item' in sft_src)

for fn in ("train_agent_sft.py", "chat_agent_sft.py"):
    src = read(fn)
    check(f"{fn} 不再包含旧标记 <|tool_call_start|>",
          "<|tool_call_start|>" not in src and "<|observation|>" not in src)
    check(f"{fn} 使用词表已有标记 <tool_call>",
          "TOOL_CALL_START = \"<tool_call>\"" in src and "OBS_START = \"<tool_response>\"" in src)
    # arguments 为 dict 时必须 json.dumps 渲染成合法 JSON（防止单引号 dict）
    check(f"{fn} 的 arguments dict 渲染为合法 JSON",
          'if isinstance(args, dict):\n                        args = json.dumps(args, ensure_ascii=False)' in src)

chat_src = read("chat_sft.py")
for token in ("<think>", "<tool_call>"):
    check(f"chat_sft.py 解码层剥离 {token} 残留", token in chat_src)

# ---------- 2. 词表与数据断言 ----------
print("== 2. 词表与数据 ==")
with open(os.path.join(BASE, "tokenizer", "tokenizer.json"), encoding="utf-8") as f:
    vocab = json.load(f)["model"]["vocab"]
required = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>"]
missing = [t for t in required if t not in vocab]
check("tokenizer 词表包含全部 6 个模板标记", not missing, f"缺失: {missing}")

with open(os.path.join(BASE, "data", "agent_sft_alpaca.jsonl"), encoding="utf-8") as f:
    first = f.read(1)
check("agent_sft_alpaca.jsonl 为真 JSONL（非数组）", first == "{", f"首字符: {first!r}")
n_agent = sum(1 for _ in open(os.path.join(BASE, "data", "agent_sft_alpaca.jsonl"), encoding="utf-8") if _.strip())
check("agent_sft_alpaca.jsonl 可逐行解析", n_agent > 0, f"行数: {n_agent}")

# ---------- 3. jinja 模板断言 ----------
print("== 3. chat_template 一致性 ==")
for d in ("pretrain_output/best_pretrain", "pretrain_output/best_sft"):
    j = read(os.path.join(d, "chat_template.jinja"))
    check(f"{d}/chat_template.jinja 无 think 块", "<think>" not in j)
    check(f"{d}/chat_template.jinja 含 <tool_call> 格式", "<tool_call>" in j)
    check(f"{d}/chat_template.jinja 含 <tool_response> 格式", "<tool_response>" in j)

with open(os.path.join(BASE, "tokenizer", "tokenizer_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
cfg_tpl = cfg.get("chat_template", "")
check("tokenizer_config.json 含 chat_template", bool(cfg_tpl))
check("tokenizer_config.json 的 chat_template 无 think 块", "<think>" not in cfg_tpl)

# ---------- 4. 参考渲染与损失掩码 ----------
print("== 4. 渲染与损失掩码 ==")
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
TOOL_CALL_START, TOOL_CALL_END = "<tool_call>", "</tool_call>"
OBS_START, OBS_END = "<tool_response>", "</tool_response>"


def render(messages):
    """与 train_sft._format_example / chat_agent_sft._render_messages 一致的规则。"""
    out = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "") or ""
        if role == "assistant":
            s = f"{IM_START}assistant\n{content}"
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                s += (f"{TOOL_CALL_START}\n"
                      f'{{"name": "{fn.get("name", "")}", '
                      f'"arguments": {fn.get("arguments", "{}")}}}\n'
                      f"{TOOL_CALL_END}")
            s += f"{IM_END}\n"
            out.append(s)
        elif role == "tool":
            out.append(f"{IM_START}user\n{OBS_START}\n{content}\n{OBS_END}{IM_END}\n")
        else:
            out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return "".join(out)


class MockTok:
    """字符级 mock：特殊 token 整体映射，其余逐字符。只用于验证边界与掩码。"""
    specials = {"<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 0,
                "<tool_call>": 21, "</tool_call>": 22,
                "<tool_response>": 23, "</tool_response>": 24}

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            hit = None
            for sp in sorted(self.specials, key=len, reverse=True):
                if text.startswith(sp, i):
                    hit = sp
                    break
            if hit:
                ids.append(self.specials[hit])
                i += len(hit)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids


def mask_labels(messages, tok):
    """与 train_sft._format_example 相同的掩码规则（无截断版本）。"""
    ids, labels = [], []
    for m in messages:
        role = m["role"]
        content = m.get("content", "") or ""
        if role == "assistant":
            body = content
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                body += (f"{TOOL_CALL_START}\n"
                         f'{{"name": "{fn.get("name", "")}", '
                         f'"arguments": {fn.get("arguments", "{}")}}}\n'
                         f"{TOOL_CALL_END}")
            body += f"{IM_END}\n"
            prefix = f"{IM_START}assistant\n"
        elif role == "tool":
            prefix, body = f"{IM_START}user\n{OBS_START}\n", f"{content}\n{OBS_END}{IM_END}\n"
        else:
            prefix, body = f"{IM_START}{role}\n{content}{IM_END}\n", ""
        p = tok.encode(prefix)
        ids.extend(p)
        labels.extend([-100] * len(p))
        if body:
            b = tok.encode(body)
            ids.extend(b)
            labels.extend(b if role == "assistant" else [-100] * len(b))
    return ids, labels


tok = MockTok()

single = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "我是助手"},
]
exp = f"{IM_START}user\n你好{IM_END}\n{IM_START}assistant\n我是助手{IM_END}\n"
check("单轮 Alpaca 渲染文本正确", render(single) == exp, f"got {render(single)!r}")

ids, labels = mask_labels(single, tok)
real = [i for i, l in zip(ids, labels) if l != -100]
exp_real = tok.encode(f"我是助手{IM_END}\n")
check("损失掩码只覆盖 assistant 内容", real == exp_real,
      f"real={real} expect={exp_real}")

multi = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "现在几点？"},
    {"role": "assistant", "content": "我不确定，需要查一下"},
    {"role": "user", "content": "<tool_response>\n14:30\n</tool_response>"},
    {"role": "assistant", "content": "现在是 14:30。"},
]
ids, labels = mask_labels(multi, tok)
first_assistant = ids.index(tok.encode(f"{IM_START}assistant\n")[0])
# 最后一个 assistant 段的 labels 必须非 -100
tail_ids, tail_labels = ids[-10:], labels[-10:]
check("多轮最后 assistant 段计入损失", any(l != -100 for l in tail_labels))

# ---------- 5. 训练/推理渲染拼接对齐 ----------
print("== 5. 推理 prompt 与训练上下文对齐 ==")
chat_src = read("chat_sft.py")
check("chat_sft.py USER_PREFIX 与训练一致",
      'USER_PREFIX = "<|im_start|>user\\n"' in chat_src)
check("chat_sft.py ASSISTANT_PREFIX 与训练一致",
      'ASSISTANT_PREFIX = "<|im_start|>assistant\\n"' in chat_src)
check("chat_sft.py GEN_PROMPT 无 think 块", "GEN_PROMPT = ASSISTANT_PREFIX" in chat_src)

agent_src = read("chat_agent_sft.py")
check("chat_agent_sft.py 解析新 <tool_call> JSON 格式",
      'r"\\s*?(.*?)\\s*?"' in agent_src and "re.escape(TOOL_CALL_START)" in agent_src)

# ---------- 6. Agent 数据实际渲染与解析 ----------
print("== 6. Agent 数据渲染与解析 ==")
agent_records = []
with open(os.path.join(BASE, "data", "agent_sft_alpaca.jsonl"), encoding="utf-8") as f:
    for line in f:
        if line.strip():
            agent_records.append(json.loads(line))
check("agent 数据可读", len(agent_records) >= 1, f"记录数: {len(agent_records)}")

first_messages = agent_records[0]["messages"]
rendered = render(first_messages)
check("agent 首条渲染含 <tool_call> JSON 块",
      '<tool_call>\n{"name": "get_system_time", "arguments": {}}\n</tool_call>' in rendered,
      rendered[:200])
check("agent 首条渲染 tool 消息为 user+<tool_response>",
      "<|im_start|>user\n<tool_response>\n14:30:22\n</tool_response><|im_end|>\n" in rendered,
      rendered[:200])
ids, labels = mask_labels(first_messages, tok)
real = [i for i, l in zip(ids, labels) if l != -100]
exp_real = tok.encode(
    "我查询一下当前时间"
    "<tool_call>\n{\"name\": \"get_system_time\", \"arguments\": {}}\n</tool_call>"
    "<|im_end|>\n当前系统时间是14:30:22。<|im_end|>\n"
)
check("agent 首条损失掩码只覆盖 assistant 段", real == exp_real,
      f"real={real} expect={exp_real}")

tool_call_re = re.compile(
    re.escape("<tool_call>") + r"\s*?(.*?)\s*?" + re.escape("</tool_call>"), re.DOTALL
)


def _parse_ref(text):
    out = []
    for m in tool_call_re.finditer(text):
        try:
            obj = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"raw": args}
        if name:
            out.append((name, args))
    return out



gen_text = "我来查一下<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}\n</tool_call>"
parsed = _parse_ref(gen_text)
check("生成文本可解析出 (工具名, 参数字典)",
      parsed == [("get_weather", {"city": "北京"})], f"got {parsed}")


gen_text2 = "我查一下<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expr\": \"1+2\"}}</tool_call>"
parsed2 = _parse_ref(gen_text2)
check("无换行变体可解析（calculator）",
      parsed2 == [("calculator", {"expr": "1+2"})], f"got {parsed2}")


gen_text3 = "试试<tool_call>这不是 JSON</tool_call>"
parsed3 = _parse_ref(gen_text3)
check("残缺标签不产生伪工具调用", parsed3 == [], f"got {parsed3}")

# ---------- 7. arguments dict 渲染为合法 JSON（训练/推理一致） ----------
print("== 7. arguments dict 渲染与解析闭环 ==")
with open(os.path.join(BASE, "data", "agent_sft_alpaca.jsonl"), encoding="utf-8") as f:
    rec = json.loads(f.readline())


def render_agent_like(messages):
    """与 chat_agent_sft._render_messages / train_agent_sft._format 一致。"""
    out = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "") or ""
        if role == "assistant":
            s = f"{IM_START}assistant\n{content}"
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                elif not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                s += (f"{TOOL_CALL_START}\n"
                      f'{{"name": "{fn.get("name", "")}", '
                      f'"arguments": {args}}}\n'
                      f"{TOOL_CALL_END}")
            s += f"{IM_END}\n"
            out.append(s)
        elif role == "tool":
            out.append(f"{IM_START}user\n{OBS_START}\n{content}\n{OBS_END}{IM_END}\n")
        else:
            out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return "".join(out)

sample_msg = [
    {"role": "user", "content": "北京天气？"},
    {"role": "assistant", "content": "我查一下",
     "tool_calls": [{"function": {"name": "get_weather",
                                   "arguments": {"city": "北京"}}}]},
]
rendered = render_agent_like(sample_msg)
m = re.search(re.escape("<tool_call>") + r"\n(.*?)\n" + re.escape("</tool_call>"), rendered, re.DOTALL)
ok_parse = False
if m:
    try:
        obj = json.loads(m.group(1).strip())
        ok_parse = (obj.get("name") == "get_weather"
                    and obj.get("arguments") == {"city": "北京"})
    except json.JSONDecodeError:
        ok_parse = False
check("arguments dict 渲染后可被 json.loads 解析",
      ok_parse, f"rendered: {rendered[:160]!r}")

print()
if FAILED:
    print(f"共 {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("全部检查通过。")
sys.exit(0)
