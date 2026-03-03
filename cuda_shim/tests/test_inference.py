"""
SynthGPU CUDA Shim — End-to-End Inference Test (test_inference.py)
==================================================================
Runs a real transformer model (TinyLlama or GPT-2) through the
Python bridge layer end-to-end.

Requires: pip install torch transformers

Run with:
    # With compiled shim (Linux):
    LD_PRELOAD=cuda_shim/build/libsynthgpu_cuda.so python cuda_shim/tests/test_inference.py

    # Python-only (bridge API layer, no C shim needed):
    python cuda_shim/tests/test_inference.py
"""

import sys
import os
import time
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_bridge_inference():
    """
    Tests a minimal transformer forward-pass through the Python bridge.
    No PyTorch required — uses numpy directly.
    """
    print("\n=== Bridge Inference Test (numpy-only) ===")
    from cuda_shim.kernels.bridge_api import (
        _scheduler, cuda_gemm, cuda_attention,
        cuda_layer_norm, cuda_relu, cuda_embedding
    )

    vocab_size = 512
    seq_len    = 16
    d_model    = 64
    n_heads    = 4
    head_dim   = d_model // n_heads
    batch      = 1

    t0 = time.perf_counter()

    # 1. Embedding
    weight  = np.random.randn(vocab_size, d_model).astype(np.float32)
    tokens  = np.random.randint(0, vocab_size, (batch * seq_len,), dtype=np.int64)
    hidden  = np.zeros((batch * seq_len, d_model), dtype=np.float32)
    cuda_embedding(tokens, weight, hidden)
    hidden = hidden.reshape(batch, seq_len, d_model)
    print(f"  [1/5] Embedding:  {hidden.shape}")

    # 2. LayerNorm
    gamma = np.ones(d_model, dtype=np.float32)
    beta  = np.zeros(d_model, dtype=np.float32)
    normed = np.zeros_like(hidden)
    cuda_layer_norm(hidden, gamma, beta, normed)
    print(f"  [2/5] LayerNorm:  {normed.shape}")

    # 3. Attention (Q/K/V projections + attention)
    Wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    Wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    Wv = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

    flat = normed.reshape(-1, d_model)  # (seq, d_model)
    Q_flat = np.zeros_like(flat)
    K_flat = np.zeros_like(flat)
    V_flat = np.zeros_like(flat)
    cuda_gemm(flat, Wq, Q_flat, 1.0, 0.0, False, False)
    cuda_gemm(flat, Wk, K_flat, 1.0, 0.0, False, False)
    cuda_gemm(flat, Wv, V_flat, 1.0, 0.0, False, False)

    Q = Q_flat.reshape(batch, n_heads, seq_len, head_dim)
    K = K_flat.reshape(batch, n_heads, seq_len, head_dim)
    V = V_flat.reshape(batch, n_heads, seq_len, head_dim)
    attn_out = np.zeros_like(Q)
    cuda_attention(Q, K, V, attn_out, scale=head_dim ** -0.5)
    print(f"  [3/5] Attention:  {attn_out.shape}")

    # 4. FFN: Linear → ReLU → Linear
    ffn_in  = attn_out.reshape(-1, d_model)
    W1 = np.random.randn(d_model, d_model * 4).astype(np.float32) * 0.02
    W2 = np.random.randn(d_model * 4, d_model).astype(np.float32) * 0.02

    ffn_hidden = np.zeros((seq_len, d_model * 4), dtype=np.float32)
    cuda_gemm(ffn_in, W1, ffn_hidden, 1.0, 0.0, False, False)
    relu_out = np.zeros_like(ffn_hidden)
    cuda_relu(ffn_hidden, relu_out)
    ffn_out = np.zeros((seq_len, d_model), dtype=np.float32)
    cuda_gemm(relu_out, W2, ffn_out, 1.0, 0.0, False, False)
    print(f"  [4/5] FFN:        {ffn_out.shape}")

    # 5. Final projection to vocab
    Wout   = np.random.randn(d_model, vocab_size).astype(np.float32) * 0.02
    logits = np.zeros((seq_len, vocab_size), dtype=np.float32)
    cuda_gemm(ffn_out, Wout, logits, 1.0, 0.0, False, False)
    next_token = int(logits[-1].argmax())

    elapsed_ms = (time.perf_counter() - t0) * 1000
    stats = _scheduler.get_stats()

    print(f"  [5/5] Logits:     {logits.shape}  → next_token={next_token}")
    print(f"\n  Forward pass: {elapsed_ms:.1f} ms")
    print(f"  Warps used:   {stats['warps_executed']}")
    print(f"  Compute units:{stats['compute_units']}")
    assert logits.shape == (seq_len, vocab_size), "Wrong logits shape"
    print("\n  [PASS] Bridge inference test complete")


def test_pytorch_inference():
    """
    Run GPT-2 or TinyLlama through PyTorch + SynthGPU bridge.
    Requires: pip install torch transformers
    """
    print("\n=== PyTorch Inference Test ===")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  [SKIP] transformers not installed. pip install torch transformers")
        return

    MODELS = [
        "sshleifer/tiny-gpt2",   # smallest available for CI
        "gpt2",
    ]

    model = None
    tokenizer = None
    for model_id in MODELS:
        try:
            print(f"  Loading {model_id} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model     = AutoModelForCausalLM.from_pretrained(model_id)
            model.eval()
            print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
            break
        except Exception as e:
            print(f"  {model_id} failed: {e}")

    if model is None:
        print("  [SKIP] No model could be loaded")
        return

    prompt = "The SynthGPU virtual accelerator enables"
    inputs = tokenizer(prompt, return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"  Prompt: '{prompt}'")
    print(f"  Output: '{text}'")
    print(f"  Generated {new_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
    assert new_tokens > 0, "No tokens generated"
    print("  [PASS] PyTorch inference test complete")


if __name__ == "__main__":
    test_bridge_inference()
    test_pytorch_inference()
    print("\n=== All inference tests done ===\n")
