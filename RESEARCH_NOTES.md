# Edge Inference Research Notes — LFM2.5-1.2B on a 2021 Consumer Laptop

**Author:** andiekobbietks (AFJ / AK Build)
**Date:** 2026-11-03
**Hardware:** ASUS VivoBook, Intel Core i5-1135G7 (4 cores / 8 threads, Tiger Lake, 2021), Intel Iris Xe Graphics (80 EUs, integrated), 8 GB DDR4, Windows 10 Home 19045
**Software stack tested:** Ollama 0.40.13, OpenVINO 2026.4.1, optimum-intel 2.2.0 (stable) + 2.3.0.dev0 (cloned from huggingface/optimum-intel), Python 3.12.10, Node 24.20.0
**Model:** LiquidAI LFM2.5-1.2B-Instruct (hybrid conv + attention, 1.2B params)

---

## What I did

1. **Hardware verification** — measured real CPU/RAM/GPU/Vulkan state on the VivoBook: 7.7 GB RAM (0.27 GB free at start), Intel Iris Xe with Vulkan 1.2.198 (conformance 1.2.6.1), AVX2 supported.
2. **Ollama path (Vulkan on iGPU)** — installed Ollama 0.40.13, set `OLLAMA_VULKAN=1` + `OLLAMA_IGPU_ENABLE=1`, imported the local GGUF (`LFM2.5-1.2B-Instruct-Q8_0.gguf`, 1.19 GB) as `lfm2.5`. Measured decode throughput with `--verbose`:
   - Q8_0 on CPU (AVX2): **28.0 tok/s**
   - Q8_0 on GPU (Vulkan, 100% offload): **14.5 tok/s** (slower than CPU)
3. **Ollama path (Q4 on CPU)** — downloaded `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf` (663 MB, from `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`), imported as `lfm2.5-q4`. Measured:
   - Q4_0 on CPU (AVX2, 100% CPU): **38.3 tok/s**
4. **OpenVINO path (pre-converted int4)** — downloaded `OpenVINO/LFM2.5-1.2B-Instruct-int4-gq-ov` (int4 group-quantized, group_size 128, NNCF). Ran with `openvino-genai` `LLMPipeline`:
   - int4 on CPU: **40.96 tok/s** (first to cross 40)
   - int4 on iGPU: **15.28 tok/s** (again slower than CPU)
5. **OpenVINO path (local conversion attempt)** — attempted `optimum-cli export openvino` (int8 and int4) on the HF model `LiquidAI/LFM2.5-1.2B-Instruct`. **Crashed with access violation (exit -1073741819)** on both stable (2.2.0) and dev (2.3.0.dev0, cloned from `huggingface/optimum-intel`). Root cause: LFM2.5 hybrid architecture (convolutions + attention) not fully supported by optimum-intel on Windows for this model.
6. **Thread tuning** — tested OMP_NUM_THREADS variants: 4 threads → 34.96 tok/s, 8 threads → 40.85 tok/s. Best: **8 threads, ~41 tok/s**.
7. **RAM / browser tabs** — measured: model footprint 819 MB (Ollama Q4) / 638 MB (OpenVINO int4). With model loaded and Chrome closed: 2.23 GB free → ~14 light Firefox tabs, ~9 medium, ~4 heavy.

## Why I did this

The claim being verified: "LFM2.5-1.2B runs at 40–70 tok/s on an Intel i5-1135G7 + Iris Xe via Vulkan, natively, no complex setup." I wanted to check whether this is reproducible on real consumer hardware (2021-era 8 GB laptop), not just asserted.

## What it shows about edge research that others haven't

Most LLM edge benchmarks are run on dedicated GPUs (RTX 4090, M-series Macs) or high-end CPUs with 32+ GB RAM. This work shows, on a **2021 8 GB consumer laptop**:

1. **CPU beats integrated GPU for small models.** Both Vulkan (Ollama) and OpenVINO on Iris Xe are ~2× slower than the same model on CPU (AVX2/OpenVINO). The common assumption "offload to iGPU = faster" does not hold for 1.2B models on Tiger Lake.
2. **The 40 tok/s threshold is only reachable with OpenVINO int4 on CPU** (40.96), not with Vulkan (38.3 max for Q4). Runtime choice matters as much as quantization.
3. **Quantization scaling is roughly linear**: Q4 ≈ 1.4× Q8 throughput on CPU.
4. **Local conversion is fragile for hybrid architectures.** Even the dev version of optimum-intel crashes (access violation) on LFM2.5 on Windows. Pre-converted models from HuggingFace (OpenVINO org) are the reliable path.
5. **NPU is not an option on Tiger Lake.** The "NPU-optimized" OpenVINO model runs on CPU here; NPU requires Core Ultra (Meteor Lake+).
6. **Practical budget**: a 0.64 GB int4 model leaves ~1.4–2.2 GB for the OS + browser. With Chrome open, only ~4–9 Firefox tabs remain viable alongside the model.

## Reproduction commands

```powershell
# Ollama (CPU, Q4)
$env:OLLAMA_NUM_GPU="0"
ollama create lfm2.5-q4 -f ModelfileQ4
ollama run lfm2.5-q4 "..." --verbose

# OpenVINO (CPU, int4, pre-converted)
pip install openvino-genai
python -c "from openvino_genai import LLMPipeline; p=LLMPipeline('ov_model','CPU'); print(p.generate('...', max_new_tokens=300))"
```

## Files referenced

- `ov_model/` — OpenVINO int4 IR (downloaded from `OpenVINO/LFM2.5-1.2B-Instruct-int4-gq-ov`)
- `ov_bench.py`, `ov_opt.py`, `ov_var.py` — benchmark scripts (in temp, not committed)
- `Modelfile`, `ModelfileQ4` — Ollama model definitions (in temp, not committed)

## Note on this fork

This fork of `huggingface/optimum-intel` exists to document the edge-inference research above. The conversion crash on LFM2.5 is an upstream issue worth reporting to the optimum-intel maintainers. No code changes are proposed here.
