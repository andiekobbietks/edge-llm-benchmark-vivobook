# Edge LLM Inference on Constrained Intel Hardware

Researching local LLM inference on an 8 GB Intel Iris Xe laptop, with a new CUDA-free workstream investigating whether LittleBit-style sub-1-bit models can be executed or approximated through OpenVINO.

> **Scope:** reproducible edge-inference experiments, not a claim that Samsung LittleBit already runs on OpenVINO or Iris Xe.

## Current machine

ASUS VivoBook X513EA · Intel Core i5-1135G7 (4C/8T) · Intel Iris Xe (80 EUs, Tiger Lake) · 8 GB RAM · Windows 10 Home 22H2 (19045).

## Existing measured baseline

The following are historical measurements from the 3 November 2026 experiment. They are not yet a statistically repeated benchmark series.

| Runtime / model | Precision | Device | Decode throughput |
|---|---:|---|---:|
| OpenVINO GenAI, LFM2.5-1.2B | INT4 group-quantised | CPU, 8 threads | **40.96 tok/s** |
| OpenVINO GenAI, LFM2.5-1.2B | INT4 group-quantised | Iris Xe GPU | 15.28 tok/s |
| Ollama 0.40.13, LFM2.5-1.2B | Q4_0 | CPU / AVX2 | 38.3 tok/s |
| Ollama 0.40.13, LFM2.5-1.2B | Q8_0 | CPU / AVX2 | 28.0 tok/s |
| Ollama 0.40.13, LFM2.5-1.2B | Q8_0 | Vulkan / Iris Xe | 14.5 tok/s |

OpenVINO CPU thread scaling: 4 threads = 34.96 tok/s; 8 threads = 40.85 tok/s; automatic = 40.96 tok/s.

**Interpretation:** on this particular Tiger Lake / 8 GB machine, CPU inference beat the tested iGPU paths for this 1.2B model. This is a device- and workload-specific observation, not a universal rule.

See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the detailed experiment record and [docs/hardware-profile.md](docs/hardware-profile.md) for the machine profile.

## New workstream: LittleBit × OpenVINO × Intel

The new branch `research/littlebit-openvino-intel` adds a CUDA-free investigation plan and reproducible utilities.

- [LittleBit/OpenVINO feasibility assessment](docs/littlebit-openvino-feasibility.md)
- [Benchmark protocol](docs/benchmark-protocol.md)
- [Known baseline results](results/baseline-results.csv)
- [Environment capture](scripts/collect_system.py)
- [OpenVINO GenAI benchmark runner](scripts/benchmark_openvino.py)

### Important technical boundary

Samsung's official LittleBit repository is a research implementation whose documented training/evaluation workflow currently assumes CUDA. Its README describes low-rank latent factorisation, binarised factors and learned scales; LittleBit-2 adds Joint-ITQ initialisation. That does **not** establish that its checkpoint format or operators are natively supported by OpenVINO.

This project therefore separates:
1. running ordinary OpenVINO models on CPU and Iris Xe;
2. testing binary/XNOR-popcount primitives as isolated microbenchmarks;
3. investigating a genuinely compatible LittleBit checkpoint/export path;
4. comparing accuracy, memory and end-to-end latency before making deployment claims.

The upstream Samsung implementation is licensed CC BY-NC 4.0. Do not copy its implementation into commercial AKCloud code or assume commercial rights. This repository's MIT licence applies only to original material in this repository; it does not relicense upstream code, models, weights or papers.

## Quick start (Windows PowerShell)

Create a clean environment and install the OpenVINO GenAI runtime appropriate to your Python version:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/collect_system.py --output results/system-profile.json
```

Run a baseline against a pre-converted OpenVINO model directory:

```powershell
python scripts/benchmark_openvino.py --model-dir .\ov_model --device CPU --threads 8 --max-new-tokens 256 --output results/openvino-cpu.jsonl
python scripts/benchmark_openvino.py --model-dir .\ov_model --device GPU --max-new-tokens 128 --output results/openvino-gpu.jsonl
```

GPU runs are expected to fail on some configurations; the runner records the exception and exits non-zero. Keep the full error output as an experiment result rather than silently changing device settings.

## Research questions

1. Can a CPU-only, CUDA-free execution path reproduce useful parts of LittleBit's factorised binary linear operation?
2. Can the Intel GPU plugin execute the relevant primitive efficiently, or does the implementation fall back to CPU?
3. What is the real memory/latency/quality trade-off against OpenVINO INT4 and GGUF Q4 baselines?
4. Can a checkpoint be exported without importing or depending on Samsung's CUDA-specific training stack?
5. What would need to change before this is a dependable runtime rather than a research demo?

## Repository map

- `docs/` — feasibility, hardware and protocol notes
- `scripts/` — environment capture and repeatable benchmark runners
- `results/` — committed baseline facts and machine-readable future runs
- `RESEARCH_NOTES.md` — original November 2026 experiment record
- `figures/` — existing benchmark charts

## Reproducibility and reporting

Every new run should record model ID/revision, model format, quantisation, OpenVINO and GenAI versions, Python version, device string, thread count, prompt, generated token count, warmups, repetitions, elapsed time, throughput, peak/available RAM and any error. Do not compare throughput figures unless generation settings and token-counting methods are aligned.

## References

- [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) — official upstream research implementation
- [OpenVINO Intel GPU plugin](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu) — Intel GPU execution backend
- [OpenVINO documentation](https://docs.openvino.ai/)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)

## Licence

See [LICENSE](LICENSE). Upstream projects, model weights and datasets retain their own licences.
