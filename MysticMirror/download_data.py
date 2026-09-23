# -*- coding: utf-8 -*-
"""
用法:
    python download_data.py                                          # 下载全部 6 个文件（全量）
    python download_data.py --files sft_t2t_mini                     # 只下载 SFT
    python download_data.py --files pretrain_t2t_mini --limit 50000  # 抽样 5 万行预训练
    python download_data.py --files sft_t2t_mini --limit 20000       # 抽样 2 万行 SFT
"""
import argparse
import os
import sys
import time
import urllib.request

BASE_URL = "https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master"
FILES = {
    "pretrain_t2t_mini": "pretrain_t2t_mini.jsonl",
    "sft_t2t_mini": "sft_t2t_mini.jsonl",
    "dpo": "dpo.jsonl",
    "rlaif": "rlaif.jsonl",
    "agent_rl": "agent_rl.jsonl",
    "agent_rl_math": "agent_rl_math.jsonl",
}

MAX_RETRIES = 3


def download_with_limit(url: str, out_path: str, limit=None):
    """流式下载；limit=None 全量，否则只保留前 limit 行（不落全量文件）。

    断点续传：目标文件已存在时，自动跳过（如需重新下载请先删除旧文件）。
    """
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        print(f"  [跳过] {out_path} 已存在，如需重新下载请先删除")
        return None
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
                if limit is None:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                    return None
                buf = b""
                n = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            f.write(line + b"\n")
                            n += 1
                            if n >= limit:
                                return n
                return n
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  [重试 {attempt}/{MAX_RETRIES}] {exc}（{wait}s 后重试）", flush=True)
                time.sleep(wait)
            else:
                print(f"  [失败] {last_exc}", flush=True)
                return None


def main():
    parser = argparse.ArgumentParser(description="下载 minimind 训练语料到 data/")
    parser.add_argument(
        "--files", nargs="*", choices=list(FILES), default=list(FILES),
        help="要下载的文件（默认全部）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="每文件最多保留的行数（流式抽样，避免全量下载）",
    )
    parser.add_argument("--out_dir", default="./data", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for key in args.files:
        url = f"{BASE_URL}/{FILES[key]}"
        out = os.path.join(args.out_dir, FILES[key])
        print(f"[下载] {FILES[key]} (limit={args.limit})", flush=True)
        try:
            n = download_with_limit(url, out, args.limit)
            if n is None and not os.path.isfile(out):
                continue
        except Exception as exc:
            print(f"  [失败] {exc}", flush=True)
            continue
        if not os.path.isfile(out):
            print(f"  [失败] 未生成文件: {out}", flush=True)
            continue
        size_mb = os.path.getsize(out) / 1e6
        note = f"，保留 {n} 行" if args.limit else ""
        print(f"  [完成] {out} ({size_mb:.1f} MB{note})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
