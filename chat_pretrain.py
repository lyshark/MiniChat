# -*- coding: utf-8 -*-
import os
import re
import torch
from transformers import AutoTokenizer

from model_chat import ChatConfig, ChatForCausalLM

model_path = r"./pretrain_output/final_pretrain"
if not os.path.isdir(model_path):
    raise FileNotFoundError(f"模型目录不存在: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer.pad_token = tokenizer.eos_token
lm_config = ChatConfig.from_pretrained(model_path)

model = ChatForCausalLM.from_pretrained(
    model_path,
    config=lm_config,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

_BUFFER_RE = re.compile(r"<\|buffer\d+\|>")

history = ""

def chat(user_input: str):
    global history
    prompt = history + user_input

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    response_tokens = outputs[0][input_len:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    response = _BUFFER_RE.sub("", response).strip()

    history += user_input + response
    return response

if __name__ == "__main__":
    while True:
        try:
            user_msg = input("请输入文本：")
        except EOFError:
            print("\n输入结束，退出。")
            break
        if user_msg.strip() == "exit":
            break
        resp = chat(user_msg)
        print(f"续写结果：{resp}")
