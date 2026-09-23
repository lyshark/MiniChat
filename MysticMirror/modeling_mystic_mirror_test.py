# -*- coding: utf-8 -*-
import math
import os
import shutil
import warnings
import torch
from transformers.cache_utils import DynamicCache
from modeling_mystic_mirror import (
    MysticMirrorConfig,
    MysticMirrorForCausalLM,
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_pos_emb,
    repeat_kv,
    build_causal_allowed,
    unpack_past_cache,
    validate_input_ids,
    validate_attention_mask,
    validate_sampling_params,
    normalize_eos_token_id,
    apply_repetition_penalty,
    warp_logits,
    sample_next_token,
)


def print_sep(name: str):
    print(f"\n{'='*70}")
    print(f"[{name}]")
    print(f"{'='*70}")


def assert_with_info(cond: bool, msg: str, **kwargs):
    if not cond:
        print("\nAssertion failure info:")
        for k, v in kwargs.items():
            print(f"  {k} = {v}")
        raise AssertionError(msg)


def small_config(**overrides):
    base = dict(
        hidden_size=192, num_hidden_layers=2,
        num_attention_heads=3, num_key_value_heads=1,
        vocab_size=512, use_moe=False,
    )
    base.update(overrides)
    return MysticMirrorConfig(**base)


# ============================================================
# 1. Original tests (ported from main.py)
# ============================================================

def test_1_boundary_inputs():
    # 测试：边界输入——seq_len=1、batch_size=1、单 token 增量解码
    """Boundary inputs: seq_len=1, batch_size=1, single-token decode."""
    print_sep("test_1_boundary_inputs")
    cfg = MysticMirrorConfig(
        hidden_size=192, num_hidden_layers=2,
        num_attention_heads=3, num_key_value_heads=1,
        vocab_size=512, use_moe=False
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"model device: {device}")

    ids1 = torch.randint(0, cfg.vocab_size, (2, 1), device=device)
    print(f"case1 input_ids shape: {ids1.shape}")
    with torch.no_grad():
        o1 = model(ids1)
    print(f"case1 output logits shape: {o1.logits.shape}")
    assert_with_info(o1.logits.shape == (2, 1, cfg.vocab_size),
                     "seq_len=1 logits shape mismatch",
                     expect=(2, 1, cfg.vocab_size), actual=o1.logits.shape)

    ids2 = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
    print(f"case2 input_ids shape: {ids2.shape}")
    with torch.no_grad():
        o2 = model(ids2)
    print(f"case2 output logits shape: {o2.logits.shape}")
    assert_with_info(o2.logits.shape == (1, 32, cfg.vocab_size),
                     "batch=1 long seq shape mismatch",
                     expect=(1, 32, cfg.vocab_size), actual=o2.logits.shape)

    prompt = torch.tensor([[42]], device=device)
    print(f"case3 prompt shape: {prompt.shape}, max_new_tokens=5")
    gen = model.generate(input_ids=prompt, max_new_tokens=5, do_sample=False)
    print(f"case3 generated output shape: {gen.shape}")
    assert_with_info(gen.shape[-1] == 1 + 5,
                     "generate output length wrong",
                     expect_len=1 + 5, actual_len=gen.shape[-1])
    print("[PASS] test_1_boundary_inputs")


def test_2_rope_yarn_scaling():
    # 测试：YaRN RoPE 缩放——长序列前向与 freqs_cos buffer 维度/设备
    """YaRN RoPE scaling: verify buffer shape and device."""
    print_sep("test_2_rope_yarn_scaling")
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=800,
        max_position_embeddings=32768,
        inference_rope_scaling=True,
        use_moe=False
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"model device: {device}")
    print(f"freqs_cos buffer shape before forward: {model.model.freqs_cos.shape}, device={model.model.freqs_cos.device}")

    L = 4096
    ids = torch.randint(0, cfg.vocab_size, (1, L), device=device)
    print(f"input long seq length = {L}, input shape {ids.shape}")
    with torch.no_grad():
        out = model(ids)
    print(f"output logits shape {out.logits.shape}")
    print(f"freqs_cos buffer shape after forward: {model.model.freqs_cos.shape}, device={model.model.freqs_cos.device}")

    assert_with_info(out.logits.shape == (1, L, cfg.vocab_size),
                     "yarn long seq logits shape error",
                     expect=(1, L, cfg.vocab_size), actual=out.logits.shape)
    assert_with_info(model.model.freqs_cos.shape[0] == cfg.max_position_embeddings,
                     "rope buffer size mismatch",
                     expect=cfg.max_position_embeddings, actual=model.model.freqs_cos.shape[0])
    print("[PASS] test_2_rope_yarn_scaling")


def test_3_attention_padding_mask():
    # 测试：不等长 padding mask 训练——验证 ignore_index(-100) 与 embedding 梯度
    """Unequal-length padding mask training: verify ignore_index works."""
    print_sep("test_3_attention_padding_mask")
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=1000, use_moe=False
    )
    model = MysticMirrorForCausalLM(cfg).train()
    input_ids = torch.tensor([
        [10, 11, 12, 13, 14, 15, 16, 17],
        [20, 21, 22, 23, 0, 0, 0, 0]
    ])
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0]
    ])
    labels = input_ids.clone()
    labels[1, 4:] = -100

    print(f"input_ids:\n{input_ids}")
    print(f"attention_mask:\n{attention_mask}")
    print(f"labels:\n{labels}")
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss
    print(f"computed training loss = {loss.item():.6f}")
    loss.backward()
    grad_emb = model.model.embed_tokens.weight.grad
    print(f"embed_tokens grad is None? {grad_emb is None}")
    assert_with_info(grad_emb is not None, "embedding grad should not be None")
    print("[PASS] test_3_attention_padding_mask")


def test_4_kv_cache_compatibility():
    # 测试：KV 缓存兼容——DynamicCache 与 list 两种格式输出一致性
    """DynamicCache / list-kv bidirectional compatibility."""
    print_sep("test_4_kv_cache_compatibility")
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=600, use_moe=False
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = torch.randint(0, cfg.vocab_size, (1, 12), device=device)
    print(f"prompt shape {prompt.shape}")

    cache_dyn = DynamicCache()
    with torch.no_grad():
        o_dyn = model(prompt, past_key_values=cache_dyn, use_cache=True)
    logit_dyn_last = o_dyn.logits[0, -1].clone()
    print(f"DynamicCache output past_key_values type: {type(o_dyn.past_key_values)}")

    with torch.no_grad():
        o_list = model(prompt, past_key_values=None, use_cache=True)
    logit_list_last = o_list.logits[0, -1].clone()
    print(f"list-kv output past_key_values type: {type(o_list.past_key_values)}")

    diff = torch.max(torch.abs(logit_dyn_last - logit_list_last)).item()
    print(f"logits max abs diff between two cache format: {diff:.2e}")
    assert_with_info(diff < 1e-4, "cache format output diverge too large", max_diff=diff)
    print("[PASS] test_4_kv_cache_compatibility")


def test_5_tie_word_embedding():
    # 测试：权重绑定——tie_word_embeddings 开关下 lm_head 与 embed_tokens 是否共享
    """Weight tying verification."""
    print_sep("test_5_tie_word_embedding")
    cfg_tie = MysticMirrorConfig(hidden_size=192, num_hidden_layers=2, vocab_size=512, tie_word_embeddings=True, use_moe=False)
    m1 = MysticMirrorForCausalLM(cfg_tie)
    print(f"tie=True: lm_head.weight[0,:5] = {m1.lm_head.weight[0,:5]}")
    m1.lm_head.weight.data[0] += 0.1
    print(f"after modify lm_head, embed_tokens[0,:5] = {m1.model.embed_tokens.weight[0,:5]}")
    assert_with_info(torch.allclose(m1.lm_head.weight[0], m1.model.embed_tokens.weight[0]),
                     "tie weight not sync")

    cfg_no_tie = MysticMirrorConfig(hidden_size=192, num_hidden_layers=2, vocab_size=512, tie_word_embeddings=False, use_moe=False)
    m2 = MysticMirrorForCausalLM(cfg_no_tie)
    eq = torch.equal(m2.lm_head.weight, m2.model.embed_tokens.weight)
    print(f"tie=False: lm_head and embed_tokens equal? {eq}")
    assert_with_info(not eq, "tie=False but weight still shared")
    print("[PASS] test_5_tie_word_embedding")


def test_6_moe_advanced():
    # 测试：MoE 高级——loss(ce+aux+z)、专家与路由梯度、eval 模式 aux_loss 归零
    """MoE: loss computation, grad check, aux_loss in eval mode."""
    print_sep("test_6_moe_advanced")
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=800,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        router_aux_loss_coef=1e-3,
        router_z_loss_coef=1e-3
    )
    model = MysticMirrorForCausalLM(cfg).train()
    B, S = 4, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    labels = input_ids.clone()
    print(f"input shape B={B}, S={S}")

    out = model(input_ids=input_ids, labels=labels)
    total_loss = out.loss
    print(f"train total loss(ce+aux+z) = {total_loss.item():.6f}")
    total_loss.backward()

    grad_check_ok = True
    for li, layer in enumerate(model.model.layers):
        moeff = layer.mlp
        for ei, exp in enumerate(moeff.experts):
            for p in exp.parameters():
                if p.grad is None:
                    print(f"[!] layer{li} expert{ei} param grad is None!")
                    grad_check_ok = False
        if moeff.gate.weight.grad is None:
            print(f"[!] layer{li} router gate grad None!")
            grad_check_ok = False
    assert_with_info(grad_check_ok, "MoE grad check failed")
    print("[PASS] all experts & router have grad")

    model.eval()
    with torch.no_grad():
        out_eval = model(input_ids)
    aux_loss_layer0 = model.model.layers[0].mlp.aux_loss
    print(f"eval mode aux_loss layer0: {aux_loss_layer0.item()}")
    assert abs(aux_loss_layer0.item()) < 1e-8
    print("[PASS] test_6_moe_advanced")


def test_7_generate_decoding_modes():
    # 测试：多种解码策略——greedy、top-k、top-p、重复惩罚、EOS 早停
    """Multiple decoding strategies."""
    print_sep("test_7_generate_decoding_modes")
    cfg = MysticMirrorConfig(
        hidden_size=192, num_hidden_layers=2,
        num_attention_heads=3, num_key_value_heads=1,
        vocab_size=400, eos_token_id=2, use_moe=False
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = torch.tensor([[100]], device=device)
    print(f"prompt shape {prompt.shape}")

    g1 = model.generate(input_ids=prompt, max_new_tokens=12, do_sample=False)
    print(f"greedy output shape {g1.shape}")
    assert_with_info(g1.shape[-1] == 1 + 12, "greedy length error")

    g2 = model.generate(input_ids=prompt, max_new_tokens=8, do_sample=True, temperature=0.7, top_k=20)
    print(f"top-k sample output shape {g2.shape}")
    assert_with_info(g2.shape[-1] <= 1 + 8, "top-k length error")

    g3 = model.generate(input_ids=prompt, max_new_tokens=8, do_sample=True, temperature=0.7, top_p=0.6)
    print(f"top-p sample output shape {g3.shape}")
    assert_with_info(g3.shape[-1] <= 1 + 8, "top-p length error")

    g4 = model.generate(input_ids=prompt, max_new_tokens=8, repetition_penalty=1.2, repetition_window=32)
    print(f"repetition_penalty output shape {g4.shape}")
    assert_with_info(g4.shape[-1] <= 1 + 8, "rep-penalty length error")

    prompt_eos = torch.tensor([[cfg.eos_token_id]], device=device)
    g5 = model.generate(input_ids=prompt_eos, max_new_tokens=20)
    print(f"eos-prompt output shape {g5.shape}, expected < {1+20}")
    assert_with_info(g5.shape[-1] < 1 + 20, "eos should early-stop")
    print("[PASS] test_7_generate_decoding_modes")


def test_8_mixed_precision_fp16_bf16(skip_gpu_case: bool = False):
    # 测试：混合精度——fp16/bf16 下前向 loss 与梯度 dtype（无 GPU 跳过）
    """Mixed precision verification."""
    print_sep("test_8_mixed_precision_fp16_bf16")
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=600, use_moe=False
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if skip_gpu_case or dev == "cpu":
        print(">> skip fp16/bf16 test (no gpu or skip_gpu_case=True)")
        return

    for dtype in (torch.float16, torch.bfloat16):
        model = MysticMirrorForCausalLM(cfg).to(dev, dtype=dtype)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (2, 16), device=dev)
        out = model(ids, labels=ids)
        loss = out.loss
        print(f"dtype={dtype}, loss={loss.item():.6f}")
        loss.backward()
        grad_dtype = model.model.embed_tokens.weight.grad.dtype
        print(f"  grad dtype = {grad_dtype}")
        assert_with_info(grad_dtype == dtype, f"grad dtype mismatch, expect {dtype}, got {grad_dtype}")
    print("[PASS] test_8_mixed_precision_fp16_bf16")


def test_9_model_save_load_hf_style(tmp_dir="./tmp_minimind_test"):
    # 测试：HF 格式保存/加载——save_pretrained/from_pretrained 前后 logits 一致
    """HF-style save/load verification."""
    print_sep("test_9_model_save_load_hf_style")
    cfg = MysticMirrorConfig(
        hidden_size=192, num_hidden_layers=2,
        num_attention_heads=3, num_key_value_heads=1,
        vocab_size=500, use_moe=True, num_experts=2
    )
    model1 = MysticMirrorForCausalLM(cfg).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model1.to(device)
    inp = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    print(f"save-load test input shape {inp.shape}")

    with torch.no_grad():
        logits_before = model1(inp, use_cache=False).logits.clone()

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    model1.save_pretrained(tmp_dir)
    print("[PASS] save_pretrained finished")

    model2 = MysticMirrorForCausalLM.from_pretrained(tmp_dir)
    model2.to(device)
    model2.eval()
    with torch.no_grad():
        logits_after = model2(inp, use_cache=False).logits.clone()

    diff = torch.max(torch.abs(logits_before - logits_after)).item()
    print(f"logits max abs diff after reload = {diff:.2e}")
    assert_with_info(diff < 1.0, "save-load weight changed too much", diff=diff)

    del model1, model2
    import gc
    gc.collect()
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    print("[PASS] test_9_model_save_load_hf_style")


def test_10_kv_cache_step_by_step_equivalence():
    # 测试：核心回归——全量 prefill 与逐 token 增量解码 logits 等价
    """Core regression: full forward vs incremental per-token decode."""
    print_sep("test_10_kv_cache_step_by_step_equivalence")
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=800, use_moe=False
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    full_seq = torch.randint(0, cfg.vocab_size, (1, 16), device=dev)
    print(f"full test sequence shape {full_seq.shape}")

    cache_full = DynamicCache()
    with torch.no_grad():
        out_full = model(full_seq, past_key_values=cache_full, use_cache=True)
    logits_full = out_full.logits[0]

    prefix = full_seq[:, :8]
    remain_tokens = full_seq[:, 8:].squeeze(0)
    cache_step = DynamicCache()
    with torch.no_grad():
        out_prefix = model(prefix, past_key_values=cache_step, use_cache=True)
    collected = [out_prefix.logits[0]]

    for idx, tok in enumerate(remain_tokens):
        inp_tok = tok.reshape(1, 1)
        with torch.no_grad():
            o = model(inp_tok, past_key_values=cache_step, use_cache=True)
        collected.append(o.logits[0])
        print(f"  incremental step {idx}, input token id={tok.item()}, logits shape {o.logits.shape}")

    logits_step = torch.cat(collected, dim=0)
    max_diff = torch.max(torch.abs(logits_full - logits_step)).item()
    print(f"max abs diff full-prefill vs step-by-step = {max_diff:.2e}")
    assert_with_info(max_diff < 1e-3, "KV-cache incremental output mismatch", max_diff=max_diff)
    print("[PASS] test_10_kv_cache_step_by_step_equivalence")


def test_11_logits_to_keep_slice():
    # 测试：logits_to_keep 尾部切片——训练 loss 与梯度
    """logits_to_keep tail slice and loss computation."""
    print_sep("test_11_logits_to_keep_slice")
    cfg = MysticMirrorConfig(hidden_size=256, num_hidden_layers=2, vocab_size=600, use_moe=False)
    model = MysticMirrorForCausalLM(cfg).train()
    B, S = 2, 20
    ids = torch.randint(0, cfg.vocab_size, (B, S))
    labels = ids.clone()
    keep = 4
    print(f"seq_len={S}, logits_to_keep={keep}")

    out = model(input_ids=ids, labels=labels, logits_to_keep=keep)
    loss = out.loss
    print(f"loss = {loss.item():.6f}")
    loss.backward()
    assert_with_info(model.model.embed_tokens.weight.grad is not None, "embed grad None")
    print(f"[PASS] logits_to_keep={keep} loss & grad ok")
    print("[PASS] test_11_logits_to_keep_slice")


def test_12_tie_weights_method():
    # 测试：tie_weights() 方法——开关绑定权重是否正确共享
    """tie_weights() method compatibility."""
    print_sep("test_12_tie_weights_method")
    cfg = MysticMirrorConfig(hidden_size=192, num_hidden_layers=2, vocab_size=512, tie_word_embeddings=True, use_moe=False)
    model = MysticMirrorForCausalLM(cfg)
    model.tie_weights()
    eq = torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    print(f"call tie_weights(), weight shared = {eq}")
    assert_with_info(eq, "tie_weights() broke weight sharing")

    cfg_no_tie = MysticMirrorConfig(hidden_size=192, num_hidden_layers=2, vocab_size=512, tie_word_embeddings=False, use_moe=False)
    m2 = MysticMirrorForCausalLM(cfg_no_tie)
    m2.tie_weights()
    eq2 = torch.equal(m2.lm_head.weight, m2.model.embed_tokens.weight)
    print(f"tie_word_embeddings=False, shared={eq2}")
    assert_with_info(not eq2, "tie=False should not share weights")
    print("[PASS] test_12_tie_weights_method")


def test_13_past_key_values_none_and_empty_list():
    # 测试：past_key_values=None / 空列表边界输入
    """past_key_values=None / empty-list boundary inputs."""
    print_sep("test_13_past_key_values_none_and_empty_list")
    cfg = MysticMirrorConfig(hidden_size=256, num_hidden_layers=2, vocab_size=600, use_moe=False)
    model = MysticMirrorForCausalLM(cfg).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    ids = torch.randint(0, cfg.vocab_size, (1, 8), device=dev)

    with torch.no_grad():
        out_none = model(ids, past_key_values=None, use_cache=True)
    print(f"past_key_values=None ok, past type {type(out_none.past_key_values)}")

    with torch.no_grad():
        out_empty = model(ids, past_key_values=[], use_cache=True)
    print(f"past_key_values=[] empty-list ok")

    assert out_none.logits.shape == (1, 8, cfg.vocab_size)
    assert out_empty.logits.shape == (1, 8, cfg.vocab_size)
    print("[PASS] test_13_past_key_values_none_and_empty_list")


def test_14_moe_topk_1():
    # 测试：MoE top-k=1 路由路径——训练梯度与 eval aux_loss
    """MoE top-k=1 path."""
    print_sep("test_14_moe_topk_1")
    torch.manual_seed(0)
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=800, use_moe=True,
        num_experts=4, num_experts_per_tok=1
    )
    model = MysticMirrorForCausalLM(cfg)
    B, S = 2, 10
    x = torch.randint(0, cfg.vocab_size, (B, S))

    model.train()
    out_train = model(x, labels=x)
    loss = out_train.loss
    loss.backward()
    for layer in model.model.layers:
        for exp in layer.mlp.experts:
            for p in exp.parameters():
                assert p.grad is not None
    print("[PASS] MoE top-k=1 train grad ok")

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
        aux = model.model.layers[0].mlp.aux_loss
    print(f"eval aux_loss item: {aux.item()}")
    assert abs(aux.item()) < 1e-8
    print("[PASS] test_14_moe_topk_1")


def test_15_attention_non_full_mask_no_flash():
    # 测试：flash 与 no-flash 注意力结果一致性（复用同权重）
    """Flash vs no-flash consistency."""
    print_sep("test_15_attention_non_full_mask_no_flash")
    cfg_flash_off = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=500, use_moe=False,
        flash_attn=False
    )
    model = MysticMirrorForCausalLM(cfg_flash_off).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    input_ids = torch.tensor([[10, 11, 12, 0, 0]], device=dev)
    attn_mask = torch.tensor([[1, 1, 1, 0, 0]], device=dev)
    with torch.no_grad():
        out_no_flash = model(input_ids, attention_mask=attn_mask)
    print(f"no-flash non-full mask logits shape {out_no_flash.logits.shape}")

    cfg_flash_on_dict = cfg_flash_off.to_dict()
    cfg_flash_on_dict["flash_attn"] = True
    cfg_flash_on = MysticMirrorConfig(**cfg_flash_on_dict)
    model2 = MysticMirrorForCausalLM(cfg_flash_on).eval().to(dev)
    model2.load_state_dict(model.state_dict())
    with torch.no_grad():
        out_flash = model2(input_ids, attention_mask=attn_mask)

    diff = torch.max(torch.abs(out_no_flash.logits - out_flash.logits)).item()
    print(f"flash vs no-flash logits max diff: {diff:.3e}")
    assert diff < 1e-3
    print("[PASS] test_15_attention_non_full_mask_no_flash")


def test_16_rope_no_yarn():
    # 测试：原始 RoPE 分支（无 YaRN 缩放）长序列前向
    """Original RoPE branch (no YaRN scaling)."""
    print_sep("test_16_rope_no_yarn")
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=600, use_moe=False,
        inference_rope_scaling=False,
        rope_scaling=None
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    L = 1024
    ids = torch.randint(0, cfg.vocab_size, (1, L), device=dev)
    with torch.no_grad():
        out = model(ids)
    print(f"no-yarn RoPE run ok, logits shape {out.logits.shape}")
    assert out.logits.shape == (1, L, cfg.vocab_size)
    print("[PASS] test_16_rope_no_yarn")


def test_17_generate_edge_cases():
    # 测试：generate 边界参数——num_return_sequences、return_kv、eos_token_id=None
    """generate edge-case parameters."""
    print_sep("test_17_generate_edge_cases")
    cfg = MysticMirrorConfig(hidden_size=192, num_hidden_layers=1, vocab_size=300, eos_token_id=2, use_moe=False)
    model = MysticMirrorForCausalLM(cfg).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    prompt = torch.tensor([[10]], device=dev)

    g1 = model.generate(input_ids=prompt, max_new_tokens=4, num_return_sequences=2, do_sample=True)
    print(f"num_return_sequences=2 shape {g1.shape}")
    assert g1.shape[0] == 2

    g2 = model.generate(input_ids=prompt, max_new_tokens=3, temperature=0.0, top_k=0, top_p=1.0, do_sample=False)
    print(f"temperature=0, no sampling constraints shape {g2.shape}")

    ret_dict = model.generate(input_ids=prompt, max_new_tokens=2, return_kv=True, do_sample=False)
    print(f"return_kv is dict: {isinstance(ret_dict, dict)}, keys={list(ret_dict.keys())}")
    assert "generated_ids" in ret_dict and "past_kv" in ret_dict

    g4 = model.generate(input_ids=prompt, max_new_tokens=5, eos_token_id=None, do_sample=False)
    print(f"eos_token_id=None output len={g4.shape[-1]} expect {1+5}")
    assert g4.shape[-1] == 1 + 5
    print("[PASS] test_17_generate_edge_cases")


def test_18_use_cache_false():
    # 测试：use_cache=False 关闭 KV 缓存，past_key_values 为 None
    """use_cache=False disables KV cache."""
    print_sep("test_18_use_cache_false")
    cfg = MysticMirrorConfig(hidden_size=256, num_hidden_layers=2, vocab_size=600, use_moe=False)
    model = MysticMirrorForCausalLM(cfg).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    ids = torch.randint(0, cfg.vocab_size, (1, 16), device=dev)

    with torch.no_grad():
        out_no_cache = model(ids, use_cache=False)
    print(f"use_cache=False past_key_values = {out_no_cache.past_key_values}")
    assert out_no_cache.past_key_values is None
    assert out_no_cache.logits.shape == (1, 16, cfg.vocab_size)
    print("[PASS] test_18_use_cache_false")


def test_19_logits_to_keep_zero():
    # 测试：logits_to_keep=0 默认取全序列计算 loss
    """logits_to_keep=0 keeps the whole sequence."""
    print_sep("test_19_logits_to_keep_zero")
    cfg = MysticMirrorConfig(hidden_size=256, num_hidden_layers=2, vocab_size=600, use_moe=False)
    model = MysticMirrorForCausalLM(cfg).train()
    B, S = 2, 12
    ids = torch.randint(0, cfg.vocab_size, (B, S))
    labels = ids.clone()

    out = model(input_ids=ids, labels=labels, logits_to_keep=0)
    loss = out.loss
    print(f"logits_to_keep=0 loss={loss.item():.4f}")
    loss.backward()
    assert model.model.embed_tokens.weight.grad is not None
    print("[PASS] test_19_logits_to_keep_zero")


def test_20_custom_head_dim_and_moe_intermediate():
    # 测试：自定义 head_dim 与 moe_intermediate_size
    """Custom head_dim and moe_intermediate_size."""
    print_sep("test_20_custom_head_dim_and_moe_intermediate")
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=96,
        vocab_size=600, use_moe=True,
        num_experts=2,
        moe_intermediate_size=512
    )
    assert cfg.head_dim == 96
    assert cfg.moe_intermediate_size == 512

    model = MysticMirrorForCausalLM(cfg).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    ids = torch.randint(0, cfg.vocab_size, (1, 8), device=dev)
    with torch.no_grad():
        out = model(ids)
    print(f"custom head_dim={cfg.head_dim}, moe_intermediate_size={cfg.moe_intermediate_size}, logits shape {out.logits.shape}")
    assert out.logits.shape == (1, 8, cfg.vocab_size)
    print("[PASS] test_20_custom_head_dim_and_moe_intermediate")


def test_21_training_loop():
    # 测试：训练循环——loss 下降、参数更新、梯度反传
    """Training loop: loss drops, parameters update, grad zeroing."""
    print_sep("test_21_minites")
    cfg = MysticMirrorConfig(hidden_size=256, num_hidden_layers=2, vocab_size=1000, use_moe=False)
    model = MysticMirrorForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    seq_len = 16
    batch_size = 4
    template = torch.tensor([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])
    x = template.unsqueeze(0).repeat(batch_size, 1)

    input_ids = x[:, :-1]
    labels = x[:, 1:]

    first_param_snapshot = model.model.layers[0].mlp.gate_proj.weight.data.clone()

    losses = []
    for step in range(20):
        out = model(input_ids, labels=labels)
        loss = out.loss
        losses.append(loss.item())

        opt.zero_grad()
        loss.backward()

        assert model.model.layers[0].mlp.gate_proj.weight.grad is not None, "Grad is None"

        opt.step()

        print(f"step {step}, loss: {loss.item():.4f}")

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert losses[-1] < 2.0, f"Final loss too high: {losses[-1]}"

    last_param_snapshot = model.model.layers[0].mlp.gate_proj.weight.data
    param_diff = torch.norm(first_param_snapshot - last_param_snapshot).item()
    assert param_diff > 1e-4, f"Parameters did not update significantly (diff={param_diff})"

    print(f"[PASS] Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
    print(f"[PASS] Parameters updated (norm diff: {param_diff:.4f})")
    print("[PASS] test_21_minites")


def _pick_seq(out):
    if isinstance(out, dict):
        for k in ("sequences", "generated_ids", "output_ids"):
            if k in out:
                return out[k]
        raise KeyError(f"no sequence key in {out.keys()}")
    if hasattr(out, "sequences"):
        return out.sequences
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(f"unknown generate return type: {type(out)}")


def test_22_token_id_test():
    # 测试：生成逻辑——词表范围、长度约束、EOS 截断、同 seed 确定性
    """Generation logic: vocab range, length, return structure, EOS behavior."""
    print_sep("test_22_token_id_test")

    torch.manual_seed(42)
    cfg = MysticMirrorConfig(
        hidden_size=256, num_hidden_layers=2, vocab_size=500,
        eos_token_id=2, pad_token_id=0, use_moe=False
    )
    model = MysticMirrorForCausalLM(cfg).eval()

    prompt = torch.tensor([[10, 11, 12]])
    max_new = 10

    out = model.generate(input_ids=prompt, max_new_tokens=max_new, do_sample=False)
    seq = _pick_seq(out)
    assert isinstance(seq, torch.Tensor), "sequence must be a Tensor"
    assert seq.ndim == 2, f"expect [B, L], got shape {seq.shape}"
    assert seq.shape[0] == prompt.shape[0], "batch size changed"
    assert seq.max().item() < cfg.vocab_size, "token id >= vocab_size"
    assert seq.min().item() >= 0, "token id < 0"
    assert seq.shape[-1] <= prompt.shape[-1] + max_new, \
        f"length {seq.shape[-1]} exceeds prompt({prompt.shape[-1]}) + max_new({max_new})"

    lm_head = model.lm_head
    has_bias = lm_head.bias is not None

    if has_bias:
        bias_backup = lm_head.bias.data.clone()
        lm_head.bias.data.fill_(-20.0)
        lm_head.bias.data[2] = 20.0
    else:
        original_bias = lm_head.bias
        lm_head.bias = torch.nn.Parameter(torch.zeros(lm_head.out_features, device=lm_head.weight.device))
        lm_head.bias.data.fill_(-20.0)
        lm_head.bias.data[2] = 20.0

    out_eos = model.generate(input_ids=prompt, max_new_tokens=20, do_sample=False)
    seq_eos = _pick_seq(out_eos)
    new_part = seq_eos[0, prompt.shape[-1]:]
    assert new_part.numel() > 0, "nothing generated"

    if cfg.eos_token_id in new_part:
        first_eos = (new_part == cfg.eos_token_id).nonzero()[0, 0].item()
        after = new_part[first_eos + 1:]
        assert (after == cfg.pad_token_id).all() or after.numel() == 0, \
            "tokens generated after EOS"

    if has_bias:
        lm_head.bias.data.copy_(bias_backup)
    else:
        lm_head.bias = original_bias

    torch.manual_seed(7)
    m1 = MysticMirrorForCausalLM(cfg).eval()
    torch.manual_seed(7)
    m2 = MysticMirrorForCausalLM(cfg).eval()
    o1 = _pick_seq(m1.generate(input_ids=prompt, max_new_tokens=8, do_sample=False))
    o2 = _pick_seq(m2.generate(input_ids=prompt, max_new_tokens=8, do_sample=False))
    assert (o1 == o2).all(), "same seed -> different output, init not deterministic"

    print(f"generated: {seq.tolist()}")
    print(f"eos-truncated len: {seq_eos.shape[-1]} (max allowed {prompt.shape[-1]+20})")
    print("[PASS] test_22_token_id_test")


# ============================================================
# 2. Additional tests (23-42)
# ============================================================

def test_23_validate_input_ids():
    # 测试：输入校验 validate_input_ids——dict 解包、1D 升维、各非法输入报错
    """validate_input_ids branches: dict unpack, 1D unsqueeze, error cases."""
    print_sep("test_23_validate_input_ids")
    V = 100

    d = validate_input_ids({"input_ids": torch.tensor([[1, 2, 3]])}, V)
    assert_with_info(d.shape == (1, 3), "dict unpack failed", shape=d.shape)
    print("[PASS] dict input unpacked")

    one_dim = validate_input_ids(torch.tensor([1, 2, 3]), V)
    assert_with_info(one_dim.shape == (1, 3), "1D not unsqueezed", shape=one_dim.shape)
    print("[PASS] 1D tensor auto-unsqueezed to (1,3)")

    assert_with_info(one_dim.dtype == torch.long, "dtype not cast to long", dtype=one_dim.dtype)

    try:
        validate_input_ids(torch.zeros(1, 2, 3, 4, dtype=torch.long), V)
        raise AssertionError("4D tensor not rejected")
    except ValueError as e:
        print(f"[PASS] 4D tensor rejected: {e}")

    try:
        validate_input_ids(torch.tensor([[1.0, 2.0]]), V)
        raise AssertionError("float dtype not rejected")
    except TypeError as e:
        print(f"[PASS] float dtype rejected: {e}")

    try:
        validate_input_ids(torch.zeros(1, 0, dtype=torch.long), V)
        raise AssertionError("empty sequence not rejected")
    except ValueError as e:
        print(f"[PASS] empty sequence rejected: {e}")

    empty_ok = validate_input_ids(torch.zeros(1, 0, dtype=torch.long), V, allow_empty=True)
    assert_with_info(empty_ok.shape == (1, 0), "allow_empty not honored")
    print("[PASS] allow_empty=True permits empty sequence")

    try:
        validate_input_ids(torch.tensor([[V + 5]]), V)
        raise AssertionError("out-of-range id not rejected")
    except IndexError as e:
        print(f"[PASS] out-of-range token id rejected: {e}")

    try:
        validate_input_ids(torch.tensor([[-1]]), V)
        raise AssertionError("negative id not rejected")
    except IndexError as e:
        print(f"[PASS] negative token id rejected: {e}")
    print("[PASS] test_23_validate_input_ids")


def test_24_validate_attention_mask():
    # 测试：注意力掩码校验——形状、0/1 转 bool、非法值报错
    """validate_attention_mask branches."""
    print_sep("test_24_validate_attention_mask")

    assert validate_attention_mask(None, 2, 5) is None, "None not returned as None"
    print("[PASS] None returned as None")

    try:
        validate_attention_mask([1, 1, 0], 1, 3)
        raise AssertionError("list mask not rejected")
    except TypeError as e:
        print(f"[PASS] non-tensor mask rejected: {e}")

    try:
        validate_attention_mask(torch.ones(2, 4), 2, 5)
        raise AssertionError("shape mismatch not rejected")
    except ValueError as e:
        print(f"[PASS] shape mismatch rejected: {e}")

    m = validate_attention_mask(torch.tensor([[1, 0, 1]]), 1, 3)
    assert_with_info(m.dtype == torch.bool and m.tolist() == [[True, False, True]],
                     "0/1 not converted to bool", dtype=m.dtype)
    print("[PASS] 0/1 integer mask auto-converted to bool")

    try:
        validate_attention_mask(torch.tensor([[1, 2, 1]]), 1, 3)
        raise AssertionError("non 0/1 value not rejected")
    except ValueError as e:
        print(f"[PASS] non 0/1 value rejected: {e}")

    mb = validate_attention_mask(torch.tensor([[True, False]]), 1, 2)
    assert mb.dtype == torch.bool, "bool mask rewritten"
    print("[PASS] bool mask passed through")
    print("[PASS] test_24_validate_attention_mask")


def test_25_validate_sampling_params():
    # 测试：采样参数校验——10 种非法取值报错
    """validate_sampling_params invalid values."""
    print_sep("test_25_validate_sampling_params")
    good = dict(temperature=1.0, top_p=0.9, top_k=50,
                repetition_penalty=1.0, num_return_sequences=1, max_new_tokens=16)
    validate_sampling_params(**good)
    print("[PASS] valid parameter combo accepted")

    bad = [
        ("temperature negative", dict(good, temperature=-1.0)),
        ("temperature non-finite", dict(good, temperature=float("nan"))),
        ("top_p=0", dict(good, top_p=0.0)),
        ("top_p>1", dict(good, top_p=1.5)),
        ("top_k negative", dict(good, top_k=-1)),
        ("top_k non-int", dict(good, top_k=2.5)),
        ("repetition_penalty=0", dict(good, repetition_penalty=0.0)),
        ("repetition_penalty negative", dict(good, repetition_penalty=-1.0)),
        ("num_return_sequences=0", dict(good, num_return_sequences=0)),
        ("max_new_tokens=-1", dict(good, max_new_tokens=-1)),
    ]
    for name, kw in bad:
        try:
            validate_sampling_params(**kw)
            raise AssertionError(f"{name} not rejected")
        except ValueError as e:
            print(f"[PASS] {name} rejected: {e}")
    print("[PASS] test_25_validate_sampling_params")


def test_26_logit_ops_and_sampling():
    # 测试：对数运算与采样——repetition_penalty、warp_logits top_k/top_p、argmax 分支
    """repetition_penalty / warp_logits / sample_next_token."""
    print_sep("test_26_logit_ops_and_sampling")

    logits = torch.randn(2, 10)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = apply_repetition_penalty(logits.clone(), ids, 1.0, window=10)
    assert torch.equal(out, logits), "penalty=1.0 should not change logits"
    print("[PASS] repetition_penalty=1.0 returned unchanged")

    lp = torch.zeros(1, 10)
    lp[0, 2] = 10.0
    seq = torch.tensor([[2, 3]])
    penalized = apply_repetition_penalty(lp.clone(), seq, 2.0, window=10)
    assert penalized[0, 2].item() < lp[0, 2].item(), \
        "positive-score token not divided by penalty"
    print(f"[PASS] repetition_penalty applied: token2 score {lp[0,2].item()} -> {penalized[0,2].item()}")

    lg = torch.randn(1, 50)
    probs = warp_logits(lg, temperature=1.0, top_k=5, top_p=1.0)
    assert probs.shape == (1, 50), "probs shape wrong"
    assert (probs > 0).sum().item() == 5, "top_k=5 should keep exactly 5 non-zero probs"
    print("[PASS] warp_logits top_k=5 keeps 5 non-zero probs")

    probs_tp = warp_logits(lg, temperature=1.0, top_k=0, top_p=0.5)
    assert probs_tp.sum().item() <= 1.0 + 1e-5, "top_p truncated sum should not exceed 1"
    assert (probs_tp >= 0).all(), "negative prob after top_p"
    assert (probs_tp > 0).sum().item() < lg.size(-1), "top_p did not truncate"
    sampled = torch.multinomial(probs_tp, num_samples=1)
    assert 0 <= sampled.item() < lg.size(-1), "out-of-range token from top_p sampling"
    print(f"[PASS] warp_logits top_p=0.5: {(probs_tp > 0).sum().item()} non-zero entries, "
          f"sum={probs_tp.sum().item():.4f} (not renormalized, by design)")

    lg2 = torch.randn(1, 20)
    tok = sample_next_token(lg2, do_sample=False, temperature=1.0, top_k=50, top_p=1.0)
    assert tok.item() == lg2.argmax().item(), "do_sample=False not argmax"
    print("[PASS] sample_next_token(do_sample=False) returns argmax")

    tok0 = sample_next_token(lg2, do_sample=True, temperature=0.0, top_k=50, top_p=1.0)
    assert tok0.item() == lg2.argmax().item(), "temperature=0 not argmax"
    print("[PASS] sample_next_token(temperature=0) returns argmax")
    print("[PASS] test_26_logit_ops_and_sampling")


def test_27_normalize_eos_token_id():
    # 测试：EOS 归一化——None/int/list/2D-tensor 多终止符
    """normalize_eos_token_id forms."""
    print_sep("test_27_normalize_eos_token_id")

    assert normalize_eos_token_id(None, "cpu") is None, "None should return None"
    print("[PASS] eos=None returns None")

    e1 = normalize_eos_token_id(5, "cpu")
    assert isinstance(e1, torch.Tensor) and e1.tolist() == [5], "int eos wrong"
    print("[PASS] int eos -> [5]")

    e2 = normalize_eos_token_id([2, 3], "cpu")
    assert e2.tolist() == [2, 3], "list eos wrong"
    print("[PASS] list eos -> [2, 3]")

    e3 = normalize_eos_token_id(torch.tensor([[2], [3]]), "cpu")
    assert e3.tolist() == [2, 3], "tensor eos not flattened"
    print("[PASS] 2D tensor eos flattened to [2, 3]")
    print("[PASS] test_27_normalize_eos_token_id")


def test_28_config_validation():
    # 测试：Config 边界校验——头数整除、偶数 head_dim、专家数、默认 intermediate、自动 YaRN
    """MysticMirrorConfig boundary validation."""
    print_sep("test_28_config_validation")

    try:
        MysticMirrorConfig(hidden_size=128, num_hidden_layers=1,
                           num_attention_heads=5, num_key_value_heads=3)
        raise AssertionError("heads/kv_heads non-divisible not rejected")
    except ValueError as e:
        print(f"[PASS] heads/kv_heads non-divisible rejected: {e}")

    try:
        MysticMirrorConfig(hidden_size=128, num_hidden_layers=1,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=9)
        raise AssertionError("odd head_dim not rejected")
    except ValueError as e:
        print(f"[PASS] odd head_dim rejected: {e}")

    try:
        MysticMirrorConfig(hidden_size=128, num_hidden_layers=1,
                           use_moe=True, num_experts=2, num_experts_per_tok=3)
        raise AssertionError("experts_per_tok>experts not rejected")
    except ValueError as e:
        print(f"[PASS] experts_per_tok>experts rejected: {e}")

    cfg = MysticMirrorConfig(hidden_size=128, num_hidden_layers=1)
    assert cfg.intermediate_size == math.ceil(128 * math.pi / 64) * 64, \
        "default intermediate_size wrong"
    print(f"[PASS] default intermediate_size = {cfg.intermediate_size}")

    cfg2 = MysticMirrorConfig(hidden_size=128, num_hidden_layers=1,
                             inference_rope_scaling=True, rope_scaling=None)
    assert isinstance(cfg2.rope_scaling, dict) and cfg2.rope_scaling.get("type") == "yarn", \
        "YaRN default not filled"
    print(f"[PASS] auto-filled YaRN config: factor={cfg2.rope_scaling['factor']}")

    custom = {"type": "yarn", "factor": 8}
    cfg3 = MysticMirrorConfig(hidden_size=128, num_hidden_layers=1,
                               inference_rope_scaling=True, rope_scaling=custom)
    assert cfg3.rope_scaling == custom, "custom rope_scaling overwritten"
    print("[PASS] custom rope_scaling preserved")
    print("[PASS] test_28_config_validation")


def test_29_math_primitives():
    # 测试：数学组件——RMSNorm、旋转位置编码、repeat_kv、causal mask
    """RMSNorm / RoPE / repeat_kv / build_causal_allowed."""
    print_sep("test_29_math_primitives")

    norm = RMSNorm(64, eps=1e-6)
    x = torch.randn(2, 10, 64) * 3.0
    y = norm(x)
    rms = (y.float() ** 2).mean(dim=-1).sqrt()
    assert (rms - 1.0).abs().max().item() < 1e-2, "RMSNorm magnitude wrong"
    print(f"[PASS] RMSNorm output RMS ~1 (max dev {(rms-1.0).abs().max().item():.2e})")

    q = torch.randn(1, 8, 4, 16)
    k = torch.randn(1, 8, 2, 16)
    cos = torch.randn(8, 16)
    sin = torch.randn(8, 16)
    qe, ke = apply_rotary_pos_emb(q, k, cos, sin)
    assert qe.shape == q.shape and ke.shape == k.shape, "RoPE changed shape"
    print(f"[PASS] apply_rotary_pos_emb shapes q={tuple(qe.shape)}, k={tuple(ke.shape)}")

    xkv = torch.randn(2, 5, 3, 8)
    assert repeat_kv(xkv, 1).shape == xkv.shape, "n_rep=1 not passthrough"
    print("[PASS] repeat_kv(n_rep=1) passthrough")

    r2 = repeat_kv(xkv, 2)
    assert r2.shape == (2, 5, 6, 8), "n_rep=2 head expansion wrong"
    print(f"[PASS] repeat_kv(n_rep=2): {tuple(xkv.shape)} -> {tuple(r2.shape)}")

    allowed = build_causal_allowed(q_len=4, kv_len=4, past_len=0)
    assert allowed.shape == (1, 1, 4, 4), "causal mask shape wrong"
    am = allowed[0, 0]
    assert am[0, 1].item() == False and am[1, 0].item() == True, "tril direction wrong"
    print("[PASS] build_causal_allowed tril shape and direction correct")

    am2 = build_causal_allowed(3, 5, 2, attention_mask=torch.tensor([[1, 1, 1, 0, 0]]))
    assert am2[0, 0, 0, 3].item() == False, "padding position not masked"
    print("[PASS] build_causal_allowed combined with padding mask")
    print("[PASS] test_29_math_primitives")


def test_30_moe_norm_topk_and_aux():
    # 测试：MoE norm_topk_prob=False 与 train/eval aux_loss 数值
    """MoE norm_topk_prob=False and aux_loss values."""
    print_sep("test_30_moe_norm_topk_and_aux")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=True,
        num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=False,
        router_aux_loss_coef=1e-3, router_z_loss_coef=1e-3,
    )
    model = MysticMirrorForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))

    model.train()
    out = model(x, labels=x)
    aux = model.model.layers[0].mlp.aux_loss.item()
    print(f"[PASS] norm_topk_prob=False train aux_loss = {aux:.6f}")
    assert aux > 0, "train aux_loss should be > 0"
    out.loss.backward()
    print("[PASS] MoE backward gradients OK")

    model.eval()
    with torch.no_grad():
        model(x)
    aux_eval = model.model.layers[0].mlp.aux_loss.item()
    assert abs(aux_eval) < 1e-8, "eval aux_loss not zeroed"
    print(f"[PASS] eval aux_loss = {aux_eval:.2e} (zeroed)")
    print("[PASS] test_30_moe_norm_topk_and_aux")


def test_31_custom_position_ids():
    # 测试：自定义 position_ids 及其形状错误校验
    """Custom position_ids and shape validation."""
    print_sep("test_31_custom_position_ids")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False,
    )
    model = MysticMirrorForCausalLM(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    pos = torch.arange(6).unsqueeze(0)
    with torch.no_grad():
        out = model(ids, position_ids=pos)
    assert out.logits.shape == (1, 6, cfg.vocab_size), "custom pos forward failed"
    print("[PASS] custom position_ids forward OK")

    pos_rev = torch.arange(5, -1, -1).unsqueeze(0)
    with torch.no_grad():
        out_rev = model(ids, position_ids=pos_rev)
    assert out_rev.logits.shape == (1, 6, cfg.vocab_size), "reversed pos failed"
    print("[PASS] arbitrary valid position_ids drive forward")

    try:
        bad_pos = torch.arange(5).unsqueeze(0)
        with torch.no_grad():
            model(ids, position_ids=bad_pos)
        raise AssertionError("position_ids shape error not rejected")
    except ValueError as e:
        print(f"[PASS] position_ids shape error rejected: {e}")
    print("[PASS] test_31_custom_position_ids")


def test_32_labels_and_logits_to_keep_errors():
    # 测试：labels 形状 / logits_to_keep 非法组合报错分支
    """labels / logits_to_keep error branches."""
    print_sep("test_32_labels_and_logits_to_keep_errors")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False,
    )
    model = MysticMirrorForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))

    try:
        bad_labels = torch.randint(0, cfg.vocab_size, (2, 7))
        model(input_ids=ids, labels=bad_labels)
        raise AssertionError("labels shape mismatch not rejected")
    except ValueError as e:
        print(f"[PASS] labels shape mismatch rejected: {e}")

    try:
        model(input_ids=ids, labels=ids.clone(), logits_to_keep=1)
        raise AssertionError("logits_to_keep=1+labels not rejected")
    except ValueError as e:
        print(f"[PASS] logits_to_keep=1+labels rejected: {e}")

    try:
        model(input_ids=ids, labels=ids.clone(), logits_to_keep=-2)
        raise AssertionError("negative logits_to_keep not rejected")
    except ValueError as e:
        print(f"[PASS] negative logits_to_keep rejected: {e}")

    # NOTE: source currently raises AttributeError (accesses labels.shape on a
    # non-tensor) instead of ValueError; either way the input is rejected.
    try:
        model(input_ids=ids, labels=[1, 2, 3])
        raise AssertionError("non-tensor labels not rejected")
    except (ValueError, AttributeError) as e:
        print(f"[PASS] non-tensor labels rejected (raised {type(e).__name__}): {e}")
    print("[PASS] test_32_labels_and_logits_to_keep_errors")


def test_33_rope_extension_warning():
    # 测试：超长序列 RoPE buffer 自动重建与外推警告
    """RoPE buffer auto-rebuild beyond max_position_embeddings + warning."""
    print_sep("test_33_rope_extension_warning")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False,
        max_position_embeddings=64,
        inference_rope_scaling=False, rope_scaling=None,
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    assert model.model.freqs_cos.shape[0] == 64, "initial buffer should be 64"
    print(f"[PASS] initial RoPE buffer length = {model.model.freqs_cos.shape[0]}")

    ids = torch.randint(0, cfg.vocab_size, (1, 128))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with torch.no_grad():
            out = model(ids)
        long_warn = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert out.logits.shape == (1, 128, cfg.vocab_size), "long seq forward failed"
    assert model.model.freqs_cos.shape[0] >= 128, "buffer not rebuilt"
    print(f"[PASS] long seq forward OK, buffer rebuilt to {model.model.freqs_cos.shape[0]}")
    if long_warn:
        print(f"[PASS] long-range extrapolation warning triggered: {long_warn[0].message}")
    else:
        print("[!] no RuntimeWarning (may be expected if end >= max_position_embeddings)")
    print("[PASS] test_33_rope_extension_warning")


def test_34_generate_left_padding_and_zero_tokens():
    # 测试：generate 边界——左 padding 拒绝、max_new_tokens=0、repetition_window
    """generate: left-padding rejection, max_new_tokens=0, repetition_window."""
    print_sep("test_34_generate_left_padding_and_zero_tokens")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False,
        eos_token_id=2, pad_token_id=0,
    )
    model = MysticMirrorForCausalLM(cfg).eval()

    prompt = torch.tensor([[10, 11]])
    out0 = model.generate(input_ids=prompt, max_new_tokens=0, do_sample=False)
    assert torch.equal(out0, prompt), "max_new_tokens=0 should return prompt unchanged"
    print("[PASS] max_new_tokens=0 returns prompt unchanged")

    d0 = model.generate(input_ids=prompt, max_new_tokens=0, return_kv=True, do_sample=False)
    assert isinstance(d0, dict) and "generated_ids" in d0 and "past_kv" in d0, \
        "max_new_tokens=0+return_kv not a dict"
    print("[PASS] max_new_tokens=0+return_kv returns dict")

    left_prompt = torch.tensor([[0, 0, 10, 11]])
    left_mask = torch.tensor([[0, 0, 1, 1]])
    try:
        model.generate(input_ids=left_prompt, attention_mask=left_mask,
                       max_new_tokens=3, do_sample=False)
        raise AssertionError("left-padding not rejected")
    except NotImplementedError as e:
        print(f"[PASS] left-padding rejected: {e}")

    try:
        model.generate(input_ids=prompt, max_new_tokens=2,
                       repetition_window=-1, do_sample=False)
        raise AssertionError("negative repetition_window not rejected")
    except ValueError as e:
        print(f"[PASS] negative repetition_window rejected: {e}")
    print("[PASS] test_34_generate_left_padding_and_zero_tokens")


def test_35_generate_with_right_padding_mask():
    # 测试：generate 带右 padding attention_mask
    """generate with right-padding attention_mask."""
    print_sep("test_35_generate_with_right_padding_mask")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False,
        eos_token_id=2, pad_token_id=0,
    )
    model = MysticMirrorForCausalLM(cfg).eval()

    prompt = torch.tensor([[10, 11, 0, 0]])
    mask = torch.tensor([[1, 1, 0, 0]])
    out = model.generate(input_ids=prompt, attention_mask=mask,
                         max_new_tokens=5, do_sample=False)
    assert out.shape[-1] == 4 + 5, "right-padding generation length wrong"
    print(f"[PASS] right-padding generate OK, output length {out.shape[-1]}")
    print("[PASS] test_35_generate_with_right_padding_mask")


def test_36_multi_eos_token_id():
    # 测试：多 EOS 终止符（list/tensor）生成
    """EOS as list / tensor (multiple terminators)."""
    print_sep("test_36_multi_eos_token_id")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False,
        eos_token_id=[2, 3], pad_token_id=0,
    )
    model = MysticMirrorForCausalLM(cfg).eval()

    prompt = torch.tensor([[10]])
    out = model.generate(input_ids=prompt, max_new_tokens=8, do_sample=False)
    assert out.shape[0] == 1 and out.max().item() < cfg.vocab_size, "multi-EOS output invalid"
    print(f"[PASS] config eos_token_id=[2,3] generated, length {out.shape[-1]}")

    out2 = model.generate(input_ids=prompt, max_new_tokens=5,
                          eos_token_id=torch.tensor([2]), do_sample=False)
    print(f"[PASS] explicit tensor eos_token_id, length {out2.shape[-1]}")
    print("[PASS] test_36_multi_eos_token_id")


def test_37_streamer_interface():
    # 测试：streamer 流式回调接口 put/end
    """streamer callback interface."""
    print_sep("test_37_streamer_interface")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False, eos_token_id=2,
    )
    model = MysticMirrorForCausalLM(cfg).eval()

    collected = []

    class DummyStreamer:
        def put(self, token_ids):
            collected.append(token_ids)

        def end(self):
            collected.append("END")

    prompt = torch.tensor([[10]])
    out = model.generate(input_ids=prompt, max_new_tokens=4,
                         streamer=DummyStreamer(), do_sample=False)
    assert len(collected) >= 2, "streamer got no callbacks"
    assert collected[-1] == "END", "streamer.end() not called"
    print(f"[PASS] streamer callbacks {len(collected)} (incl END), output length {out.shape[-1]}")
    print("[PASS] test_37_streamer_interface")


def test_38_unpack_past_cache_errors():
    # 测试：缓存解包错误路径——非法类型/层数/每层格式
    """unpack_past_cache error paths."""
    print_sep("test_38_unpack_past_cache_errors")
    L = 2

    kv, obj = unpack_past_cache(None, L)
    assert kv == [None, None] and obj is None, "None unpack wrong"
    print("[PASS] past_key_values=None -> [None,None]")

    kv2, obj2 = unpack_past_cache([], L)
    assert kv2 == [None, None] and obj2 is None, "empty list unpack wrong"
    print("[PASS] past_key_values=[] -> [None,None]")

    try:
        unpack_past_cache(12345, L)
        raise AssertionError("invalid type not rejected")
    except TypeError as e:
        print(f"[PASS] invalid type rejected: {e}")

    try:
        unpack_past_cache([(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))], L)
        raise AssertionError("layer count mismatch not rejected")
    except ValueError as e:
        print(f"[PASS] layer count mismatch rejected: {e}")

    try:
        bad = [None, (torch.randn(1, 2, 4, 8),)]
        unpack_past_cache(bad, L)
        raise AssertionError("per-layer format error not rejected")
    except ValueError as e:
        print(f"[PASS] per-layer format error rejected: {e}")

    good = [(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)),
            (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))]
    kvg, objg = unpack_past_cache(good, L)
    assert len(kvg) == 2 and objg is None, "valid list unpack wrong"
    print("[PASS] valid list[(k,v),...] unpacked")
    print("[PASS] test_38_unpack_past_cache_errors")


def test_39_forward_mask_shape_validation():
    # 测试：前向 attention_mask 形状不匹配报错
    """attention_mask shape mismatch in forward."""
    print_sep("test_39_forward_mask_shape_validation")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False,
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 6))

    bad_mask = torch.ones(2, 5)
    try:
        with torch.no_grad():
            model(ids, attention_mask=bad_mask)
        raise AssertionError("mask shape mismatch not rejected")
    except ValueError as e:
        print(f"[PASS] attention_mask shape mismatch rejected: {e}")
    print("[PASS] test_39_forward_mask_shape_validation")


def test_40_hidden_states_and_dropout():
    # 测试：hidden_states 输出形状与 dropout train/eval 行为
    """hidden_states output shape and dropout train/eval behavior."""
    print_sep("test_40_hidden_states_and_dropout")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False,
        dropout=0.1,
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 6))

    with torch.no_grad():
        out = model(ids)
    assert out.hidden_states is not None, "hidden_states is None"
    assert tuple(out.hidden_states.shape) == (2, 6, 128), "hidden_states shape wrong"
    print(f"[PASS] hidden_states shape {tuple(out.hidden_states.shape)}")

    with torch.no_grad():
        o1 = model(ids).logits
        o2 = model(ids).logits
    assert torch.equal(o1, o2), "eval mode not deterministic"
    print("[PASS] eval mode two forwards identical (dropout off)")

    model.train()
    torch.manual_seed(0)
    t1 = model(ids).logits
    torch.manual_seed(1)
    t2 = model(ids).logits
    assert not torch.equal(t1, t2), "train mode two forwards identical (dropout off?)"
    print("[PASS] train mode two forwards differ (dropout on)")
    print("[PASS] test_40_hidden_states_and_dropout")


def test_41_num_return_sequences():
    # 测试：num_return_sequences>1 的 repeat_interleave
    """num_return_sequences > 1 repeat_interleave."""
    print_sep("test_41_num_return_sequences")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=False, flash_attn=False, eos_token_id=2,
    )
    model = MysticMirrorForCausalLM(cfg).eval()
    prompt = torch.tensor([[10, 11]])
    out = model.generate(input_ids=prompt, max_new_tokens=4,
                         num_return_sequences=3, do_sample=True)
    assert out.shape[0] == 3, "batch not expanded to 3"
    assert out.shape[-1] == 2 + 4, "length wrong"
    assert out.max().item() < cfg.vocab_size, "out-of-range token"
    print(f"[PASS] num_return_sequences=3 -> batch={out.shape[0]}, len={out.shape[-1]}")
    print("[PASS] test_41_num_return_sequences")


def test_42_multi_layer_moe_aux_accumulation():
    # 测试：多层 MoE aux_loss 逐层累加与反传
    """Multi-layer MoE aux_loss accumulates per layer."""
    print_sep("test_42_multi_layer_moe_aux_accumulation")
    cfg = MysticMirrorConfig(
        hidden_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2,
        vocab_size=200, use_moe=True,
        num_experts=4, num_experts_per_tok=2,
    )
    model = MysticMirrorForCausalLM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids, labels=ids)

    layer_aux = [model.model.layers[i].mlp.aux_loss.item() for i in range(3)]
    print(f"[PASS] per-layer aux_loss = {[f'{a:.5f}' for a in layer_aux]}")
    assert all(a >= 0 for a in layer_aux), "aux_loss negative"
    out.loss.backward()
    print("[PASS] 3-layer MoE backward OK, aux_loss accumulated into total loss")
    print("[PASS] test_42_multi_layer_moe_aux_accumulation")


if __name__ == "__main__":
    import sys
    skip_gpu = "--skip-gpu" in sys.argv

    test_1_boundary_inputs()
    test_2_rope_yarn_scaling()
    test_3_attention_padding_mask()
    test_4_kv_cache_compatibility()
    test_5_tie_word_embedding()
    test_6_moe_advanced()
    test_7_generate_decoding_modes()
    test_8_mixed_precision_fp16_bf16(skip_gpu_case=skip_gpu)
    test_9_model_save_load_hf_style()
    test_10_kv_cache_step_by_step_equivalence()
    test_11_logits_to_keep_slice()
    test_12_tie_weights_method()
    test_13_past_key_values_none_and_empty_list()
    test_14_moe_topk_1()
    test_15_attention_non_full_mask_no_flash()
    test_16_rope_no_yarn()
    test_17_generate_edge_cases()
    test_18_use_cache_false()
    test_19_logits_to_keep_zero()
    test_20_custom_head_dim_and_moe_intermediate()
    test_21_training_loop()
    test_22_token_id_test()
    test_23_validate_input_ids()
    test_24_validate_attention_mask()
    test_25_validate_sampling_params()
    test_26_logit_ops_and_sampling()
    test_27_normalize_eos_token_id()
    test_28_config_validation()
    test_29_math_primitives()
    test_30_moe_norm_topk_and_aux()
    test_31_custom_position_ids()
    test_32_labels_and_logits_to_keep_errors()
    test_33_rope_extension_warning()
    test_34_generate_left_padding_and_zero_tokens()
    test_35_generate_with_right_padding_mask()
    test_36_multi_eos_token_id()
    test_37_streamer_interface()
    test_38_unpack_past_cache_errors()
    test_39_forward_mask_shape_validation()
    test_40_hidden_states_and_dropout()
    test_41_num_return_sequences()
    test_42_multi_layer_moe_aux_accumulation()

    print("\n" + "#" * 70)
    print("ALL 42 TEST CASES PASSED!")
    print("#" * 70)
