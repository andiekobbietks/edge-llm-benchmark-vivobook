# Benchmark protocol

## Rules

1. Run at least one warmup, then at least five measured repetitions for a reportable result.
2. Keep prompt text, max-new-token count, sampling settings, model revision and device settings fixed across comparisons.
3. Report median and spread (minimum/maximum or interquartile range), not only the fastest run.
4. Separate model load time, first-token latency and steady-state decode throughput.
5. Record RAM before load, after load and after generation. GPU memory statistics may be unavailable; report that explicitly.
6. Preserve stderr and full exception text for failed runs.
7. Do not compare a kernel microbenchmark with end-to-end tokens/second.

## Minimum metadata

- Timestamp (local and UTC if possible)
- Host profile and driver
- Python, OpenVINO and OpenVINO GenAI versions
- Model repository, revision, format and quantisation
- Device and thread count
- Prompt and generation configuration
- Warmup count, repetition count and actual output token count
- Load time, first-token latency, total generation time and decode tok/s
- RAM/available memory and any error/fallback information

## Commands

```powershell
python scripts/collect_system.py --output results/system-profile.json
python scripts/benchmark_openvino.py --model-dir .\ov_model --device CPU --threads 8 --warmup 1 --repeats 5 --max-new-tokens 256 --output results/openvino-cpu.jsonl
python scripts/benchmark_openvino.py --model-dir .\ov_model --device GPU --warmup 1 --repeats 3 --max-new-tokens 128 --output results/openvino-gpu.jsonl
```

A GPU failure does not invalidate the CPU baseline. Keep separate result files and do not overwrite earlier runs.
