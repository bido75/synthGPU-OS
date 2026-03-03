"""
SynthGPU CUDA Shim — PyTorch Compatibility Test Suite (test_pytorch.py)
=======================================================================
Full PyTorch compatibility tests covering device detection, tensor ops,
nn layers, and model inference.

Run with:
    # Linux (with compiled shim):
    LD_PRELOAD=cuda_shim/build/libsynthgpu_cuda.so python cuda_shim/tests/test_pytorch.py

    # Python-only (no compiled shim needed for many tests):
    python cuda_shim/tests/test_pytorch.py
"""

import sys
import os
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LD_PRELOAD = os.environ.get("LD_PRELOAD", "")
if "synthgpu" not in LD_PRELOAD.lower():
    print("WARNING: LD_PRELOAD not set to SynthGPU shim.")
    print("  Linux: export LD_PRELOAD=cuda_shim/build/libsynthgpu_cuda.so")
    print("  Some device-detection tests will be skipped.\n")

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed. Install with: pip install torch")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []

def test(name, fn, require_shim=False):
    if require_shim and "synthgpu" not in LD_PRELOAD.lower():
        print(f"  [{SKIP}] {name}  (requires LD_PRELOAD)")
        results.append((name, None))
        return
    if not HAS_TORCH:
        print(f"  [{SKIP}] {name}  (no PyTorch)")
        results.append((name, None))
        return
    try:
        fn()
        print(f"  [{PASS}] {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e}")
        results.append((name, False))

# ── Group 1: Device Detection ─────────────────────────────────────
print("\n[1] Device Detection (requires LD_PRELOAD on Linux)")

test("cuda.is_available() == True",
     lambda: _assert(torch.cuda.is_available()),
     require_shim=True)

test("cuda.device_count() == 1",
     lambda: _assert(torch.cuda.device_count() == 1),
     require_shim=True)

test("cuda.get_device_name(0) contains 'SynthGPU'",
     lambda: _assert("SynthGPU" in torch.cuda.get_device_name(0)),
     require_shim=True)

test("cuda.get_device_properties(0).total_memory > 0",
     lambda: _assert(torch.cuda.get_device_properties(0).total_memory > 0),
     require_shim=True)

test("cuda.current_device() == 0",
     lambda: _assert(torch.cuda.current_device() == 0),
     require_shim=True)

# ── Group 2: Tensor creation ──────────────────────────────────────
print("\n[2] Tensor Creation and Movement")

test("torch.zeros().cuda()",
     lambda: _assert(torch.zeros(10, 10).cuda().device.type == "cuda"),
     require_shim=True)

test("torch.ones().cuda().sum() == 25",
     lambda: _assert(torch.ones(5, 5).cuda().sum().item() == 25.0),
     require_shim=True)

test("tensor.to('cuda')",
     lambda: _assert(torch.randn(10).to("cuda").device.type == "cuda"),
     require_shim=True)

test("tensor.cpu() round-trip preserves values", lambda: (
    _check_roundtrip()
), require_shim=True)

def _check_roundtrip():
    original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    cuda_t   = original.cuda()
    back     = cuda_t.cpu()
    assert torch.allclose(original, back, atol=1e-5), "Round-trip mismatch"

# ── Group 3: Math operations ──────────────────────────────────────
print("\n[3] Math Operations on CUDA Tensors")

test("torch.matmul CUDA — identity check",
     lambda: _check_matmul(),
     require_shim=True)

def _check_matmul():
    A = torch.ones(4, 4).cuda()
    B = torch.ones(4, 4).cuda() * 2
    C = torch.matmul(A, B)
    assert C.shape == (4, 4)
    assert abs(C[0, 0].item() - 8.0) < 1e-3, f"Expected 8.0 got {C[0,0].item()}"

test("torch.mm CUDA shape check",
     lambda: _assert(torch.mm(torch.ones(3, 4).cuda(), torch.ones(4, 5).cuda()).shape == (3, 5)),
     require_shim=True)

test("torch.bmm batched matmul",
     lambda: _assert(torch.bmm(torch.ones(2, 3, 4).cuda(), torch.ones(2, 4, 5).cuda()).shape == (2, 3, 5)),
     require_shim=True)

test("torch.relu negative → zero",
     lambda: _assert(torch.relu(torch.tensor([-1., 0., 1.]).cuda()).tolist() == [0., 0., 1.]),
     require_shim=True)

test("torch.softmax sums to 1",
     lambda: _assert(abs(torch.softmax(torch.ones(5).cuda(), 0).sum().item() - 1.0) < 1e-5),
     require_shim=True)

test("torch.sum CUDA",
     lambda: _assert(torch.ones(100).cuda().sum().item() == 100.0),
     require_shim=True)

test("torch.mean CUDA",
     lambda: _assert(abs(torch.ones(10).cuda().mean().item() - 1.0) < 1e-5),
     require_shim=True)

test("torch.max CUDA",
     lambda: _assert(torch.tensor([1., 3., 2.]).cuda().max().item() == 3.0),
     require_shim=True)

# ── Group 4: Neural Network layers ────────────────────────────────
print("\n[4] Neural Network Layers")

test("nn.Linear forward pass",
     lambda: _check_linear(),
     require_shim=True)

def _check_linear():
    layer = torch.nn.Linear(64, 32).cuda()
    x     = torch.randn(8, 64).cuda()
    out   = layer(x)
    assert out.shape == (8, 32), f"Wrong shape: {out.shape}"

test("nn.LayerNorm forward pass",
     lambda: _check_layernorm(),
     require_shim=True)

def _check_layernorm():
    ln  = torch.nn.LayerNorm(64).cuda()
    x   = torch.randn(4, 64).cuda()
    out = ln(x)
    assert out.shape == (4, 64)

test("nn.Embedding forward pass",
     lambda: _check_embedding(),
     require_shim=True)

def _check_embedding():
    emb = torch.nn.Embedding(1000, 64).cuda()
    ids = torch.randint(0, 1000, (8, 16)).cuda()
    out = emb(ids)
    assert out.shape == (8, 16, 64)

test("nn.MultiheadAttention forward",
     lambda: _check_attention(),
     require_shim=True)

def _check_attention():
    mha = torch.nn.MultiheadAttention(64, 8, batch_first=True).cuda()
    x   = torch.randn(2, 10, 64).cuda()
    out, _ = mha(x, x, x)
    assert out.shape == (2, 10, 64)

test("nn.Softmax normalises",
     lambda: _assert(abs(torch.nn.Softmax(dim=0)(torch.ones(4).cuda()).sum().item() - 1.0) < 1e-5),
     require_shim=True)

# ── Group 5: Model-level ──────────────────────────────────────────
print("\n[5] Model-Level Integration")

test("3-layer MLP forward pass",
     lambda: _check_mlp(),
     require_shim=True)

def _check_mlp():
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.LayerNorm(128),
        nn.Linear(128, 10),
    ).cuda()
    x   = torch.randn(16, 128).cuda()
    out = model(x)
    assert out.shape == (16, 10)

test("GPT-2 forward pass (skipped if transformers not installed)",
     lambda: _check_gpt2())

def _check_gpt2():
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        model     = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        inputs    = tokenizer("Hello world", return_tensors="pt")
        inputs    = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        assert output.logits.shape[-1] == 50257
    except ImportError:
        print("    (transformers not installed — skipping GPT-2 test)")

# ── Summary ───────────────────────────────────────────────────────
def _assert(condition):
    if not condition:
        raise AssertionError("Assertion failed")

total  = len(results)
passed = sum(1 for _, ok in results if ok is True)
failed = sum(1 for _, ok in results if ok is False)
skipped = sum(1 for _, ok in results if ok is None)

print(f"\n{'='*60}")
print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped  (total {total})")
if failed:
    print("  Failed:")
    for name, ok in results:
        if ok is False:
            print(f"    FAIL  {name}")
print(f"{'='*60}\n")

if __name__ == "__main__":
    sys.exit(0 if failed == 0 else 1)
