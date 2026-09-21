import json
import re

def jsonl_to_txt(jsonl_path, txt_path, max_lines=None, start_line=0, clean_mode=True):
    write_count = 0
    keep_pattern = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9，.]")
    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(txt_path, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            raw_line = line.strip()
            if not raw_line:
                continue
            if idx < start_line:
                continue
            try:
                data = json.loads(raw_line)
                text_content = data.get("text", "")
                if not text_content or not text_content.strip():
                    continue
                if clean_mode:
                    text_content = text_content.replace("\n", " ").replace("\r", " ")
                    clean_text = keep_pattern.sub("", text_content)
                    clean_text = clean_text.strip()
                    if not clean_text:
                        continue
                    output_text = clean_text
                else:
                    output_text = text_content.rstrip("\n")

                f_out.write(output_text + "\n")
                write_count += 1
                if max_lines is not None and write_count >= max_lines:
                    break
            except json.JSONDecodeError:
                print(f"[警告]行{idx} JSON解析失败，已跳过")
    print(f"处理完成，成功写入 {write_count} 条文本")

if __name__ == "__main__":
    INPUT_JSONL = "./data/pretrain_t2t_mini.jsonl"
    OUTPUT_TXT = "./data/train_data.txt"
    START_LINE = 0
    MAX_LINES = 9999999999
    CLEAN_MODE = True
    jsonl_to_txt(
        jsonl_path=INPUT_JSONL,
        txt_path=OUTPUT_TXT,
        max_lines=MAX_LINES,
        start_line=START_LINE,
        clean_mode=CLEAN_MODE
    )