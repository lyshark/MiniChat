# -*- coding: utf-8 -*-
import json
import random

SRC = r"E:\MiniChat\MysticMirror\data\sft_t2t_mini.jsonl"
DST = r"E:\MiniChat\MysticMirror\data\dpo_smoke.jsonl"

REJECT_POOL = [
    "我不知道。",
    "抱歉，我无法回答这个问题。",
    "这个问题太简单了，我不想回答。",
    "不清楚，你去问别人吧。",
]

out = []
with open(SRC, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        convs = item.get("conversations", [])
        user_turn = next((m for m in convs if m.get("role") == "user"), None)
        asst_turn = next((m for m in convs if m.get("role") == "assistant"), None)
        if user_turn and asst_turn and asst_turn.get("content", "").strip():
            out.append({
                "prompt": user_turn["content"],
                "chosen": asst_turn["content"],
                "rejected": random.choice(REJECT_POOL),
            })
        if len(out) >= 60:
            break

with open(DST, "w", encoding="utf-8") as f:
    for item in out:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"写出 {len(out)} 条 -> {DST}")
