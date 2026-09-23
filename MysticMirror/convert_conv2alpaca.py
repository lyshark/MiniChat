# -*- coding: utf-8 -*-
"""
转换 conversations 格式(jsonl) → Alpaca instruction‑output 格式(jsonl)
只提取每一条样本里第一轮：user → assistant，丢弃后续多轮对话
用法：
    python convert_conv2alpaca.py --in_file input.jsonl --out_file output.jsonl --limit 100
    # limit 代表只取前多少行原始数据
"""
import argparse
import json

def convert_one_item(raw_item):
    convs = raw_item.get("conversations", [])
    user_msg = None
    assistant_msg = None
    for msg in convs:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role == "user" and user_msg is None:
            user_msg = content
        elif role == "assistant" and assistant_msg is None:
            assistant_msg = content
        if user_msg is not None and assistant_msg is not None:
            break
    if not user_msg or not assistant_msg:
        return None
    return {
        "instruction": user_msg,
        "output": assistant_msg
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", required=True, help="输入jsonl，每条带conversations字段")
    parser.add_argument("--out_file", required=True, help="输出alpaca格式jsonl")
    parser.add_argument("--limit", type=int, default=None, help="只读取前N行原始数据，不填读取全部")
    args = parser.parse_args()

    out_records = []
    read_cnt = 0
    skip_cnt = 0

    with open(args.in_file, "r", encoding="utf‑8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and read_cnt >= args.limit:
                break
            read_cnt +=1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skip_cnt +=1
                continue
            res = convert_one_item(obj)
            if res is not None:
                out_records.append(res)
            else:
                skip_cnt +=1

    with open(args.out_file, "w", encoding="utf‑8") as f_out:
        for rec in out_records:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"读取原始行数: {read_cnt}")
    print(f"成功转换: {len(out_records)}")
    print(f"跳过(解析失败/缺少user‑assistant对): {skip_cnt}")
    print(f"输出保存至: {args.out_file}")

if __name__ == "__main__":
    main()
