玄鉴（Mystic Mirror LLM）是一套基于 PyTorch 实现的 Decoder‑Only 因果大模型网络本体，参考主流 LLaMA 技术路线，完整实现模型骨干组件，同时支持稠密 Transformer 与 MoE 混合专家两种架构。本代码仅聚焦神经网络本身，实现模型前向计算、损失函数、基础自回归生成逻辑，不包含分词器、数据集处理、分布式训练调度、高性能推理调度引擎，可对接 HuggingFace 生态完成模型训练与基础推理。


## 模型本体测试结果

```bash
======================================================================
[test_1_boundary_inputs]
======================================================================
model device: cpu
case1 input_ids shape: torch.Size([2, 1])
case1 output logits shape: torch.Size([2, 1, 512])
case2 input_ids shape: torch.Size([1, 32])
case2 output logits shape: torch.Size([1, 32, 512])
case3 prompt shape: torch.Size([1, 1]), max_new_tokens=5
case3 generated output shape: torch.Size([1, 6])
[PASS] test_1_boundary_inputs

======================================================================
[test_2_rope_yarn_scaling]
======================================================================
model device: cpu
freqs_cos buffer shape before forward: torch.Size([32768, 64]), device=cpu
input long seq length = 4096, input shape torch.Size([1, 4096])
output logits shape torch.Size([1, 4096, 800])
freqs_cos buffer shape after forward: torch.Size([32768, 64]), device=cpu
[PASS] test_2_rope_yarn_scaling

======================================================================
[test_3_attention_padding_mask]
======================================================================
input_ids:
tensor([[10, 11, 12, 13, 14, 15, 16, 17],
        [20, 21, 22, 23,  0,  0,  0,  0]])
attention_mask:
tensor([[1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0]])
labels:
tensor([[  10,   11,   12,   13,   14,   15,   16,   17],
        [  20,   21,   22,   23, -100, -100, -100, -100]])
computed training loss = 6.731242
embed_tokens grad is None? False
[PASS] test_3_attention_padding_mask

======================================================================
[test_4_kv_cache_compatibility]
======================================================================
prompt shape torch.Size([1, 12])
DynamicCache output past_key_values type: <class 'transformers.cache_utils.DynamicCache'>
list-kv output past_key_values type: <class 'list'>
logits max abs diff between two cache format: 0.00e+00
[PASS] test_4_kv_cache_compatibility

======================================================================
[test_5_tie_word_embedding]
======================================================================
tie=True: lm_head.weight[0,:5] = tensor([ 0.0008, -0.0145, -0.0037,  0.0246,  0.0029], grad_fn=<SliceBackward0>)
after modify lm_head, embed_tokens[0,:5] = tensor([0.1008, 0.0855, 0.0963, 0.1246, 0.1029], grad_fn=<SliceBackward0>)
tie=False: lm_head and embed_tokens equal? False
[PASS] test_5_tie_word_embedding

======================================================================
[test_6_moe_advanced]
======================================================================
input shape B=4, S=16
train total loss(ce+aux+z) = 6.731487
[PASS] all experts & router have grad
eval mode aux_loss layer0: 0.0
[PASS] test_6_moe_advanced

======================================================================
[test_7_generate_decoding_modes]
======================================================================
prompt shape torch.Size([1, 1])
greedy output shape torch.Size([1, 13])
top-k sample output shape torch.Size([1, 9])
top-p sample output shape torch.Size([1, 9])
repetition_penalty output shape torch.Size([1, 9])
eos-prompt output shape torch.Size([1, 1]), expected < 21
[PASS] test_7_generate_decoding_modes

======================================================================
[test_8_mixed_precision_fp16_bf16]
======================================================================
>> skip fp16/bf16 test (no gpu or skip_gpu_case=True)

======================================================================
[test_9_model_save_load_hf_style]
======================================================================
save-load test input shape torch.Size([1, 8])
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████
███████████████████████████████████████████████████████| 1/1 [00:00<00:00, 73.12it/s]
[PASS] save_pretrained finished
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████
██████████████████████████████████████████████████████| 32/32 [00:00<00:00, 3383.70it/s]
logits max abs diff after reload = 0.00e+00
[PASS] test_9_model_save_load_hf_style

======================================================================
[test_10_kv_cache_step_by_step_equivalence]
======================================================================
full test sequence shape torch.Size([1, 16])
  incremental step 0, input token id=18, logits shape torch.Size([1, 1, 800])
  incremental step 1, input token id=722, logits shape torch.Size([1, 1, 800])
  incremental step 2, input token id=41, logits shape torch.Size([1, 1, 800])
  incremental step 3, input token id=227, logits shape torch.Size([1, 1, 800])
  incremental step 4, input token id=91, logits shape torch.Size([1, 1, 800])
  incremental step 5, input token id=454, logits shape torch.Size([1, 1, 800])
  incremental step 6, input token id=712, logits shape torch.Size([1, 1, 800])
  incremental step 7, input token id=583, logits shape torch.Size([1, 1, 800])
max abs diff full-prefill vs step-by-step = 9.54e-07
[PASS] test_10_kv_cache_step_by_step_equivalence

======================================================================
[test_11_logits_to_keep_slice]
======================================================================
seq_len=20, logits_to_keep=4
loss = 6.352053
[PASS] logits_to_keep=4 loss & grad ok
[PASS] test_11_logits_to_keep_slice

======================================================================
[test_12_tie_weights_method]
======================================================================
call tie_weights(), weight shared = True
tie_word_embeddings=False, shared=False
[PASS] test_12_tie_weights_method

======================================================================
[test_13_past_key_values_none_and_empty_list]
======================================================================
past_key_values=None ok, past type <class 'list'>
past_key_values=[] empty-list ok
[PASS] test_13_past_key_values_none_and_empty_list

======================================================================
[test_14_moe_topk_1]
======================================================================
[PASS] MoE top-k=1 train grad ok
eval aux_loss item: 0.0
[PASS] test_14_moe_topk_1

======================================================================
[test_15_attention_non_full_mask_no_flash]
======================================================================
no-flash non-full mask logits shape torch.Size([1, 5, 500])
flash vs no-flash logits max diff: 5.364e-07
[PASS] test_15_attention_non_full_mask_no_flash

======================================================================
[test_16_rope_no_yarn]
======================================================================
no-yarn RoPE run ok, logits shape torch.Size([1, 1024, 600])
[PASS] test_16_rope_no_yarn

======================================================================
[test_17_generate_edge_cases]
======================================================================
num_return_sequences=2 shape torch.Size([2, 5])
temperature=0, no sampling constraints shape torch.Size([1, 4])
return_kv is dict: True, keys=['generated_ids', 'past_kv']
eos_token_id=None output len=6 expect 6
[PASS] test_17_generate_edge_cases

======================================================================
[test_18_use_cache_false]
======================================================================
use_cache=False past_key_values = None
[PASS] test_18_use_cache_false

======================================================================
[test_19_logits_to_keep_zero]
======================================================================
logits_to_keep=0 loss=6.5736
[PASS] test_19_logits_to_keep_zero

======================================================================
[test_20_custom_head_dim_and_moe_intermediate]
======================================================================
custom head_dim=96, moe_intermediate_size=512, logits shape torch.Size([1, 8, 600])
[PASS] test_20_custom_head_dim_and_moe_intermediate

======================================================================
[test_21_minites]
======================================================================
step 0, loss: 6.9440
step 1, loss: 4.8753
step 2, loss: 3.6879
step 3, loss: 2.9489
step 4, loss: 2.2793
step 5, loss: 1.8152
step 6, loss: 1.4206
step 7, loss: 1.1400
step 8, loss: 0.9259
step 9, loss: 0.7511
step 10, loss: 0.6075
step 11, loss: 0.4912
step 12, loss: 0.3980
step 13, loss: 0.3233
step 14, loss: 0.2633
step 15, loss: 0.2152
step 16, loss: 0.1765
step 17, loss: 0.1456
step 18, loss: 0.1209
step 19, loss: 0.1013
[PASS] Loss decreased from 6.9440 to 0.1013
[PASS] Parameters updated (norm diff: 2.7410)
[PASS] test_21_minites

======================================================================
[test_22_token_id_test]
======================================================================
generated: [[10, 11, 12, 2]]
eos-truncated len: 4 (max allowed 23)
[PASS] test_22_token_id_test

======================================================================
[test_23_validate_input_ids]
======================================================================
[PASS] dict input unpacked
[PASS] 1D tensor auto-unsqueezed to (1,3)
[PASS] 4D tensor rejected: input_ids 必须是 1D/2D 张量，实际维度为 4
[PASS] float dtype rejected: input_ids 必须是整数 dtype，实际为 torch.float32
[PASS] empty sequence rejected: input_ids 序列长度为 0（空 prompt），至少需要一个 token
[PASS] allow_empty=True permits empty sequence
[PASS] out-of-range token id rejected: token id 越界：input_ids 取值区间 [105, 105]，合法区间 [0, 99]
[PASS] negative token id rejected: token id 越界：input_ids 取值区间 [-1, -1]，合法区间 [0, 99]
[PASS] test_23_validate_input_ids

======================================================================
[test_24_validate_attention_mask]
======================================================================
[PASS] None returned as None
[PASS] non-tensor mask rejected: attention_mask 必须是 torch.Tensor
[PASS] shape mismatch rejected: attention_mask 形状 (2, 4) 与期望 (2, 5) 不符（长度须等于 历史缓存+当前序列）
[PASS] 0/1 integer mask auto-converted to bool
[PASS] non 0/1 value rejected: attention_mask 只能取 0/1（或 bool）
[PASS] bool mask passed through
[PASS] test_24_validate_attention_mask

======================================================================
[test_25_validate_sampling_params]
======================================================================
[PASS] valid parameter combo accepted
[PASS] temperature negative rejected: temperature 必须是非负有限实数，得到 -1.0
[PASS] temperature non-finite rejected: temperature 必须是非负有限实数，得到 nan
[PASS] top_p=0 rejected: top_p 必须落在 (0, 1]，得到 0.0
[PASS] top_p>1 rejected: top_p 必须落在 (0, 1]，得到 1.5
[PASS] top_k negative rejected: top_k 必须是非负整数，得到 -1
[PASS] top_k non-int rejected: top_k 必须是非负整数，得到 2.5
[PASS] repetition_penalty=0 rejected: repetition_penalty 必须是正实数，得到 0.0
[PASS] repetition_penalty negative rejected: repetition_penalty 必须是正实数，得到 -1.0
[PASS] num_return_sequences=0 rejected: num_return_sequences 必须是 >=1 的整数，得到 0
[PASS] max_new_tokens=-1 rejected: max_new_tokens 必须是非负整数，得到 -1
[PASS] test_25_validate_sampling_params

======================================================================
[test_26_logit_ops_and_sampling]
======================================================================
[PASS] repetition_penalty=1.0 returned unchanged
[PASS] repetition_penalty applied: token2 score 10.0 -> 5.0
[PASS] warp_logits top_k=5 keeps 5 non-zero probs
[PASS] warp_logits top_p=0.5: 10 non-zero entries, sum=0.5160 (not renormalized, by design)
[PASS] sample_next_token(do_sample=False) returns argmax
[PASS] sample_next_token(temperature=0) returns argmax
[PASS] test_26_logit_ops_and_sampling

======================================================================
[test_27_normalize_eos_token_id]
======================================================================
[PASS] eos=None returns None
[PASS] int eos -> [5]
[PASS] list eos -> [2, 3]
[PASS] 2D tensor eos flattened to [2, 3]
[PASS] test_27_normalize_eos_token_id

======================================================================
[test_28_config_validation]
======================================================================
[PASS] heads/kv_heads non-divisible rejected: num_attention_heads(5) 必须能被 num_key_value_heads(3) 整除
[PASS] odd head_dim rejected: head_dim(9) 必须为偶数（RoPE 需要成对的维度）
[PASS] experts_per_tok>experts rejected: num_experts_per_tok(3) 不能超过 num_experts(2)
[PASS] default intermediate_size = 448
[PASS] auto-filled YaRN config: factor=16
[PASS] custom rope_scaling preserved
[PASS] test_28_config_validation

======================================================================
[test_29_math_primitives]
======================================================================
[PASS] RMSNorm output RMS ~1 (max dev 1.19e-07)
[PASS] apply_rotary_pos_emb shapes q=(1, 8, 4, 16), k=(1, 8, 2, 16)
[PASS] repeat_kv(n_rep=1) passthrough
[PASS] repeat_kv(n_rep=2): (2, 5, 3, 8) -> (2, 5, 6, 8)
[PASS] build_causal_allowed tril shape and direction correct
[PASS] build_causal_allowed combined with padding mask
[PASS] test_29_math_primitives

======================================================================
[test_30_moe_norm_topk_and_aux]
======================================================================
[PASS] norm_topk_prob=False train aux_loss = 0.002922
[PASS] MoE backward gradients OK
[PASS] eval aux_loss = 0.00e+00 (zeroed)
[PASS] test_30_moe_norm_topk_and_aux

======================================================================
[test_31_custom_position_ids]
======================================================================
[PASS] custom position_ids forward OK
[PASS] arbitrary valid position_ids drive forward
[PASS] position_ids shape error rejected: position_ids 形状须为 (1, 6)，实际 (1, 5)
[PASS] test_31_custom_position_ids

======================================================================
[test_32_labels_and_logits_to_keep_errors]
======================================================================
[PASS] labels shape mismatch rejected: labels 形状须与 input_ids (2, 8) 一致，实际 (2, 7)
[PASS] logits_to_keep=1+labels rejected: 训练（labels 非空）时 logits_to_keep 必须为 0（全量）或 >=2，否则 shift 后没有可计算损失的位置
[PASS] negative logits_to_keep rejected: logits_to_keep 必须是非负整数，得到 -2
[PASS] non-tensor labels rejected (raised AttributeError): 'list' object has no attribute 'shape'
[PASS] test_32_labels_and_logits_to_keep_errors

======================================================================
[test_33_rope_extension_warning]
======================================================================
[PASS] initial RoPE buffer length = 64
[PASS] long seq forward OK, buffer rebuilt to 128
[PASS] long-range extrapolation warning triggered: 序列长度 128 超过 max_position_embeddings=64，且未配置 RoPE 缩放；RoPE 将直接外推，长程位置可能退化（建议启用 YaRN 或调大 max_position_embeddings）
[PASS] test_33_rope_extension_warning

======================================================================
[test_34_generate_left_padding_and_zero_tokens]
======================================================================
[PASS] max_new_tokens=0 returns prompt unchanged
[PASS] max_new_tokens=0+return_kv returns dict
[PASS] left-padding rejected: 检测到左 padding（行首 mask=0）；本引擎仅支持右 padding，请改用右 padding 或在 tokenizer 侧设置 padding_side='right'
[PASS] negative repetition_window rejected: repetition_window 必须非负，得到 -1
[PASS] test_34_generate_left_padding_and_zero_tokens

======================================================================
[test_35_generate_with_right_padding_mask]
======================================================================
[PASS] right-padding generate OK, output length 9
[PASS] test_35_generate_with_right_padding_mask

======================================================================
[test_36_multi_eos_token_id]
======================================================================
[PASS] config eos_token_id=[2,3] generated, length 9
[PASS] explicit tensor eos_token_id, length 6
[PASS] test_36_multi_eos_token_id

======================================================================
[test_37_streamer_interface]
======================================================================
[PASS] streamer callbacks 6 (incl END), output length 5
[PASS] test_37_streamer_interface

======================================================================
[test_38_unpack_past_cache_errors]
======================================================================
[PASS] past_key_values=None -> [None,None]
[PASS] past_key_values=[] -> [None,None]
[PASS] invalid type rejected: object of type 'int' has no len()
[PASS] layer count mismatch rejected: past_key_values 层数 1 与模型层数 2 不一致
[PASS] per-layer format error rejected: past_key_values 每层必须是 None 或 (k, v) 二元组
[PASS] valid list[(k,v),...] unpacked
[PASS] test_38_unpack_past_cache_errors

======================================================================
[test_39_forward_mask_shape_validation]
======================================================================
[PASS] attention_mask shape mismatch rejected: attention_mask 形状 (2, 5) 与期望 (2, 6) 不符（长度须等于 历史缓存+当前序列）
[PASS] test_39_forward_mask_shape_validation

======================================================================
[test_40_hidden_states_and_dropout]
======================================================================
[PASS] hidden_states shape (2, 6, 128)
[PASS] eval mode two forwards identical (dropout off)
[PASS] train mode two forwards differ (dropout on)
[PASS] test_40_hidden_states_and_dropout

======================================================================
[test_41_num_return_sequences]
======================================================================
[PASS] num_return_sequences=3 -> batch=3, len=6
[PASS] test_41_num_return_sequences

======================================================================
[test_42_multi_layer_moe_aux_accumulation]
======================================================================
[PASS] per-layer aux_loss = ['0.00252', '0.00255', '0.00259']
[PASS] 3-layer MoE backward OK, aux_loss accumulated into total loss
[PASS] test_42_multi_layer_moe_aux_accumulation

######################################################################
ALL 42 TEST CASES PASSED!
######################################################################
```
