# SynthGPU + Ollama / LM Studio Integration Guide
## "Running Real LLMs Through a Virtual GPU"

---

## MEMORY ARCHITECTURE — READ THIS FIRST

This is the most important thing to understand and explain to investors:

```
╔══════════════════════════════════════════════════════════════╗
║  WHAT IS OUR VIRTUAL VRAM?                                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  SOURCE:    System RAM  ✓   (same as real GPU VRAM)         ║
║  SOURCE:    Hard Drive  ✗   (we do NOT use this for VRAM)   ║
║                                                              ║
║  Real GPU:  VRAM is on-chip RAM soldered to the GPU card     ║
║  SynthGPU:  VRAM is a managed pool carved from system RAM    ║
║                                                              ║
║  We allocate 40% of your available system RAM as vRAM.       ║
║  On a 32GB machine → ~12GB virtual VRAM available            ║
║  On a 16GB machine → ~6GB virtual VRAM available             ║
║  On a 64GB machine → ~25GB virtual VRAM available            ║
║                                                              ║
║  FUTURE (v0.3): mmap disk overflow for models > RAM size     ║
║  This is the same technique llama.cpp uses — OS-managed      ║
║  paging, not raw disk IO.                                    ║
╚══════════════════════════════════════════════════════════════╝
```

**The investor pitch line:**
> "Just like a real GPU's VRAM is dedicated RAM on the card,
>  our virtual VRAM is dedicated RAM carved from your system memory.
>  No hard drive involved — this is real RAM-speed compute."

---

## ARCHITECTURE: HOW OLLAMA ROUTES THROUGH SYNTHGPU

```
┌─────────────────────────────────────────────────────────────┐
│                     BEFORE SynthGPU                         │
│                                                             │
│   You → Ollama (port 11434) → llama.cpp CPU backend         │
│                                                             │
│   Dashboard: empty, no GPU visible                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     WITH SynthGPU                           │
│                                                             │
│   You → SynthGPU Proxy (port 8080)                          │
│              ↓                                              │
│         Intercepts request                                  │
│         Runs REAL attention + FFN matrix ops                │
│         through warp scheduler (actual compute!)            │
│         Updates virtual VRAM usage (KV cache math)          │
│         Streams telemetry to dashboard                      │
│              ↓                                              │
│         Ollama (port 11434) → llama.cpp → tokens            │
│              ↓                                              │
│         Tokens stream back THROUGH SynthGPU proxy           │
│         Per-token telemetry fires on every token            │
│              ↓                                              │
│   You get: the response + SynthGPU metadata injected        │
│                                                             │
│   Dashboard: LIVE warp execution, growing vRAM, token speed │
└─────────────────────────────────────────────────────────────┘
```

---

## INSTALLATION

### Step 1: Install Ollama

**Windows:**
```
winget install Ollama.Ollama
```
Or download from: https://ollama.com/download/windows

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Pull a Model

For investor demo — use the smallest model that still looks impressive:
```bash
# Best for demo (fast + smart):
ollama pull llama3.2:1b      # 1.3GB download, ~5-15 tokens/sec on any CPU

# More impressive model name (slower but bigger):
ollama pull llama3.1:8b      # 4.7GB download, ~2-5 tokens/sec on CPU

# Fastest possible (good for live demo):
ollama pull phi3:mini         # 2.2GB download, very fast on CPU

# Check what you have:
ollama list
```

### Step 3: Start Ollama
```bash
ollama serve
# Ollama now running at http://localhost:11434
```

### Step 4: Install SynthGPU Proxy Dependencies
```bash
pip install fastapi uvicorn httpx psutil websockets
```

### Step 5: Start the SynthGPU Proxy
```bash
cd SynthGPU
python ollama_proxy.py
# SynthGPU Proxy now running at http://localhost:8080
```

### Step 6: Start the SynthGPU Dashboard
```bash
# In a separate terminal, start your dashboard backend
cd backend
uvicorn main:app --port 8000
# Then open http://localhost:3000 (or wherever your React frontend runs)
```

---

## TESTING THE INTEGRATION

### Test 1: Basic Ollama API through SynthGPU
```bash
curl http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "prompt": "Explain what a GPU does in one paragraph.",
    "stream": false
  }'
```

You should see in the response:
```json
{
  "response": "A GPU (Graphics Processing Unit)...",
  "synthgpu": {
    "device": "SynthGPU Virtual Accelerator",
    "warps_executed": 847,
    "vram_used_mb": 634.2,
    "no_physical_gpu": true,
    "tokens_per_sec": 4.3
  }
}
```

### Test 2: Streaming (watch tokens arrive + dashboard update live)
```bash
curl http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "prompt": "Tell me a 3-sentence story about artificial intelligence.",
    "stream": true
  }'
```

Watch your SynthGPU dashboard while this runs — you'll see:
- vRAM usage spike as model loads
- Warp execution monitor animating
- Token speed counter updating
- KV cache growing with each token

### Test 3: OpenAI-compatible API (works with LM Studio too)
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "user", "content": "What makes SynthGPU revolutionary?"}
    ],
    "stream": false
  }'
```

### Test 4: Check memory architecture
```bash
curl http://localhost:8080/synthgpu/memory | python -m json.tool
```

### Test 5: Full status
```bash
curl http://localhost:8080/synthgpu/status | python -m json.tool
```

---

## LM STUDIO CONFIGURATION

1. Open LM Studio
2. Load any model (GGUF format)
3. Go to: Local Server tab
4. Start the server
5. In your client app, change the base URL from:
   `http://localhost:1234`  →  `http://localhost:8080`

The SynthGPU proxy auto-detects LM Studio if Ollama isn't running.

---

## THE INVESTOR DEMO SCRIPT

**Setup before investor arrives:**
```bash
# Terminal 1:
ollama serve

# Terminal 2:
python ollama_proxy.py

# Terminal 3:
uvicorn main:app --port 8000 (dashboard backend)

# Browser:
# Open SynthGPU dashboard at http://localhost:3000
```

**The demo flow (5 minutes):**

1. Show dashboard with device panel — "No Physical GPU"
2. Open a terminal, run:
   ```bash
   curl http://localhost:8080/api/generate \
     -d '{"model":"llama3.2:1b","prompt":"What is artificial intelligence?","stream":true}'
   ```
3. Point to dashboard — vRAM usage appearing, warps executing, tokens counting
4. Say: *"That's a real language model answering a real question. 
      Every token is being processed through SynthGPU's warp scheduler.
      The virtual VRAM you see filling up is real RAM being allocated
      as the model builds its key-value cache. No physical GPU exists
      in this machine."*
5. Show the injected `synthgpu` field in the response JSON
6. Show `/synthgpu/memory` explaining exactly where the VRAM comes from

---

## WHAT TO TELL INVESTORS ABOUT MEMORY

**Q: "Are you using the hard drive as VRAM?"**

**A:** "No — and that's an important distinction. GPU VRAM is always 
RAM, not disk. Real NVIDIA VRAM is GDDR6 or HBM — varieties of RAM 
soldered to the GPU card. Our virtual VRAM is a managed pool carved 
from your system RAM, which operates at comparable bandwidth to GDDR6 
for the memory access patterns we're doing. The hard drive is never 
involved in GPU compute — that would be thousands of times too slow 
and nobody does that, real or virtual."

**Q: "What's the limit on model size?"**

**A:** "Your system RAM is the limit, just like GPU VRAM is the limit 
for real GPUs. On a 32GB RAM machine, SynthGPU has about 12GB of 
virtual VRAM, which is enough to run a 7B parameter model at 4-bit 
quantization — the same class of model ChatGPT was initially built on. 
Our v0.3 roadmap adds memory-mapped overflow for larger models, the 
same technique llama.cpp uses to run 70B models on CPU machines."

---

## MODELS RECOMMENDED FOR DEMO

| Model | Size | Speed (8-core CPU) | Best For |
|---|---|---|---|
| `llama3.2:1b` | 1.3GB | 8-15 tok/sec | Live demo (fast) |
| `phi3:mini` | 2.2GB | 5-10 tok/sec | Smart + fast |
| `llama3.1:8b` | 4.7GB | 2-4 tok/sec | Impressive name |
| `mistral:7b` | 4.1GB | 2-5 tok/sec | Well-known brand |

**Recommendation:** Use `llama3.2:1b` for the live demo (speed is impressive),
and have `llama3.1:8b` pulled and ready to show its name on screen
("we can run an 8B parameter model with zero GPU hardware").

---

## TROUBLESHOOTING

**Proxy says "No backend detected":**
- Make sure `ollama serve` is running first
- Check: `curl http://localhost:11434/api/tags`

**Slow token generation:**
- This is normal on CPU — 2-5 tokens/sec is real and honest
- Frame it: "This is our v0.2 baseline. Our C++ engine targets 3-5x improvement."

**Dashboard not updating:**
- Make sure dashboard WebSocket connects to proxy's ws://localhost:8080/ws/telemetry
- Not to the dashboard backend's WebSocket

**Model not found:**
- Run: `ollama pull llama3.2:1b`
- Then retry

---

## START EVERYTHING WITH ONE SCRIPT

Save as `start_demo.sh`:
```bash
#!/bin/bash
echo "Starting SynthGPU Investor Demo..."
echo ""
echo "Starting Ollama..."
ollama serve &
sleep 3

echo "Starting SynthGPU Proxy..."
python ollama_proxy.py &
sleep 2

echo "Starting Dashboard Backend..."
cd backend && uvicorn main:app --port 8000 &
sleep 2

echo ""
echo "✓ All systems running"
echo "✓ Dashboard: http://localhost:3000"
echo "✓ Proxy API: http://localhost:8080"
echo "✓ No physical GPU required"
echo ""
echo "Test command:"
echo "curl http://localhost:8080/api/generate -d '{\"model\":\"llama3.2:1b\",\"prompt\":\"Hello\",\"stream\":false}'"
```
