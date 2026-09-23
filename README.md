玄鉴（Mystic Mirror LLM）是一套基于 PyTorch 实现的 Decoder‑Only 因果大模型网络本体，参考主流 LLaMA 技术路线，完整实现模型骨干组件，同时支持稠密 Transformer 与 MoE 混合专家两种架构。本代码仅聚焦神经网络本身，实现模型前向计算、损失函数、基础自回归生成逻辑，不包含分词器、数据集处理、分布式训练调度、高性能推理调度引擎，可对接 HuggingFace 生态完成模型训练与基础推理。

一款自主知识产权的大模型完整源码，由LYSHARK开发维护。

## 预训练

准备数据集，可参考目录下的`MysticMirror\data\train_data.jsonl`格式来清洗数据集。也可以下载minimind的数据集来直接测试。

```json
{"text": "给我生成一首有关秋天的诗歌秋日早晨"}
{"text": "根据以下输入的问题，生成一句话回答你觉得寿司好不好吃作为一名AI"}
```

单卡启动，最终的最优模型会存储到`./pretrain_output/best_pretrain`目录下

```bash
# 单卡训练
(torch) root@pomldvfkqabpefgl-hunt-bdcdb8fdd-6fgbd:/data/MysticMirror# ll
total 320
drwxr-xr-x 9 root root  4096 Sep 23 12:59 ./
drwxr-xr-x 8 root root   137 Sep 23 12:55 ../
drwxr-xr-x 2 root root    52 Sep 23 12:59 __pycache__/
drwxrwxrwx 2 root root     6 Sep 23 09:51 agent_sft_output/
-rwxrwxrwx 1 root root 23001 Sep 23 12:13 chat_agent_sft.py*
-rwxrwxrwx 1 root root 13571 Sep 23 12:27 chat_dpo.py*
-rwxrwxrwx 1 root root 10827 Sep 23 08:02 chat_pretrain.py*
-rwxrwxrwx 1 root root 14159 Sep 23 11:27 chat_sft.py*
drwxrwxrwx 2 root root     6 Sep 22 21:25 checkpoint/
drwxrwxrwx 2 root root    63 Sep 23 12:56 data/
-rwxrwxrwx 1 root root  4524 Sep 23 11:46 download_data.py*
-rwxrwxrwx 1 root root  2670 Sep 23 12:35 dpo_verify_preference.py*
-rwxrwxrwx 1 root root  1399 Sep 23 12:30 make_dpo_smoke.py*
-rwxrwxrwx 1 root root 38144 Sep 22 23:20 modeling_mystic_mirror.py*
-rwxrwxrwx 1 root root 63567 Sep 22 16:51 modeling_mystic_mirror_test.py*
drwxrwxrwx 2 root root     6 Sep 23 12:45 pretrain_output/
drwxrwxrwx 2 root root     6 Sep 23 12:45 smoke_test/
-rwxrwxrwx 1 root root 12382 Sep 23 12:29 test_dpo.py*
drwxrwxrwx 2 root root    84 Sep 23 11:16 tokenizer/
-rwxrwxrwx 1 root root 21071 Sep 23 12:12 train_agent_sft.py*
-rwxrwxrwx 1 root root 29409 Sep 23 12:29 train_dpo.py*
-rwxrwxrwx 1 root root 16276 Sep 23 12:14 train_pretrain.py*
-rwxrwxrwx 1 root root 23624 Sep 23 12:10 train_sft.py*
-rwxrwxrwx 1 root root 14319 Sep 23 12:15 verify_templates.py*

# 切割出前40万行
(torch) root@pomldvfkqabpefgl-hunt-bdcdb8fdd-6fgbd:/data/MysticMirror/data# cat pretrain_t2t_mini.jsonl | wc -l
1270238
(torch) root@pomldvfkqabpefgl-hunt-bdcdb8fdd-6fgbd:/data/MysticMirror/data# head -n 200000 pretrain_t2t_mini.jsonl > pretrain_data_20K.jsonl
(torch) root@pomldvfkqabpefgl-hunt-bdcdb8fdd-6fgbd:/data/MysticMirror/data# ls -lh
total 3.7G
-rw-r--r-- 1 root root 187M Sep 23 13:24 pretrain_data.jsonl
-rw-r--r-- 1 root root 110M Sep 23 13:28 pretrain_data_20K.jsonl
-rw-r--r-- 1 root root 1.2G May 25 15:47 pretrain_t2t_mini.jsonl
-rw-r--r-- 1 root root 567M Sep 23 13:20 sft_alpaca_40K.jsonl
-rw-r--r-- 1 root root 1.7G May 24 09:21 sft_t2t_mini.jsonl

# 开始训练
python train_pretrain.py --data_path ./data/pretrain_data_20K.jsonl --tokenizer_path ./tokenizer/ --output_dir ./pretrain_output/ --vocab_size 6400 --hidden_size 512 --num_hidden_layers 8 --num_attention_heads 8 --num_key_value_heads 4 --max_position_embeddings 2048 --max_seq_len 2048 --batch_size 8 --gradient_accumulation_steps 4 --learning_rate 3e-4 --warmup_steps 100 --num_train_epochs 1 --eval_steps 500 --save_steps 100 --mixed_precision bf16

# 恢复继续训练
python train_pretrain.py --data_path ./data/pretrain_data_20K.jsonl --tokenizer_path ./tokenizer/ --output_dir ./pretrain_output/ --vocab_size 6400 --hidden_size 512 --num_hidden_layers 8 --num_attention_heads 8 --num_key_value_heads 4 --max_position_embeddings 2048 --max_seq_len 2048 --batch_size 8 --gradient_accumulation_steps 4 --learning_rate 3e-4 --warmup_steps 100 --num_train_epochs 1 --eval_steps 500 --save_steps 100 --mixed_precision bf16 --resume_from_checkpoint ./pretrain_output/checkpoint-3000
```

测试数据是否可用
```bash
python chat_pretrain.py --model_path ./pretrain_output/best_pretrain

加载分词器：./pretrain_output/best_pretrain
加载模型配置：./pretrain_output/best_pretrain
加载模型权重（device=cuda, dtype=torch.bfloat16）：./pretrain_output/best_pretrain
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90/90 [00:00<00:00, 3686.65it/s]
模型加载完成：参数量 30,025,216 | vocab=6400 | layers=8 | max_position_embeddings=2048

输入 exit / quit 退出；直接回车表示基于已有历史继续续写。

请输入文本：你好
续写结果：，我听说您是我们公司的创始人之一，可以介绍一下吗小李当然，我们是一家大型企业，主要业务包括电子商务和游戏市场，以推广电子产品和营销活动为主打，同时也是我们公司的核心领袖之一小李听起来很有前途，但我还想问一下您对这方面的看法小李我们的团队一直在努力开发和创新，但是对于企业的整体发展还有待改进小李非常好，我们有很多创新的想法，不过我们还需要更多的研发团队来推动我们的创新，您有什么具体的计划或目标吗小李我们正在开发一个在线购物平台，该平台将根据不同的需求提供个性化的产品和服务，并且可以为我们的网站提供更好的体验

请输入文本：
```

# SFT微调

准备数据集格式如下：

```json
{"instruction": "你模型的训练数据来源是什么？", "output": "我的训练数据涵盖多领域，确保覆盖广泛，但具体细节不公开。"}
{"instruction": "训练数据的来源和多样性如何？", "output": "我的训练数据来源广泛，涵盖多个领域"}
{"instruction": "训练数据如何保障信息准确性？", "output": "我的训练数据通过多轮验证和质量控制流程进行保障，确保信息的准确性和可靠性。"}
```

- `./sft_output/best_sft`：验证集 loss 最低的最优模型（优先用这个做推理）
- `./sft_output/final_sft`：最后一轮结束的模型

使用参数训练

我抽取了40万行，大概训练一小时。

```bash
python convert_conv2alpaca.py --in_file ./data/sft_t2t_mini.jsonl --out_file ./data/sft_alpaca_40K.jsonl --limit 400000
读取原始行数: 400000
成功转换: 399999
跳过(解析失败/缺少user‑assistant对): 1
输出保存至: ./data/sft_alpaca_40K.jsonl

python train_sft.py --pretrain_model_dir /data/MysticMirror/pretrain_output/best_pretrain --data_path ./data/sft_alpaca_40K.jsonl --output_dir ./sft_output --max_seq_len 2048 --batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-5 --weight_decay 0.0 --warmup_steps 100 --num_train_epochs 3 --max_grad_norm 1.0 --eval_steps 100 --save_steps 500 --seed 42 --mixed_precision bf16
```

对话测试
```bash
python chat_sft.py --model_path ./pretrain_output/best_sft
```

## Agent SFT

```bash
python train_agent_sft.py --base_model_dir ./pretrain_output/best_sft --data_path ./data/agent_sft_alpaca.jsonl --num_train_epochs 5 --batch_size 4

# 训练 Agent SFT（在装有模型的机器上）
python train_agent_sft.py --base_model_dir ./sft_output/best_sft --data_path ./data/agent_sft_alpaca.jsonl

# 对话测试
python chat_agent_sft.py --model_path ./agent_sft_output/best_agent_sft
python chat_agent_sft.py --model_path ./sft_output/best_sft --device cuda
```
