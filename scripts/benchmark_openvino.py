#!/usr/bin/env python3
"""Repeatable OpenVINO GenAI benchmark runner for CPU or Intel GPU."""
from __future__ import annotations
import argparse, json, statistics, time, traceback
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PROMPT = "Explain why measuring CPU and integrated GPU inference separately matters for local language models."

def token_count(pipe, text: str) -> int:
    encoded = pipe.get_tokenizer().encode(text)
    ids = getattr(encoded, "input_ids", encoded)
    try:
        return len(ids[0]) if len(ids) and hasattr(ids[0], "__len__") else len(ids)
    except Exception:
        return len(str(text).split())

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads; 0 leaves OpenVINO defaults")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", default="results/openvino-run.jsonl")
    args = ap.parse_args()
    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": args.model_dir, "device": args.device, "threads": args.threads or "runtime-default",
        "max_new_tokens": args.max_new_tokens, "warmup_runs": args.warmup, "repeats": args.repeats,
        "prompt": args.prompt,
    }
    try:
        import openvino as ov
        import openvino_genai as ov_genai
        record["openvino_version"] = ov.__version__
        record["openvino_genai_version"] = getattr(ov_genai, "__version__", "unknown")
        start_load = time.perf_counter()
        pipe = ov_genai.LLMPipeline(args.model_dir, args.device)
        record["load_seconds"] = time.perf_counter() - start_load
        if args.warmup:
            for _ in range(args.warmup):
                pipe.generate(args.prompt, max_new_tokens=min(16, args.max_new_tokens))
        timings, counts = [], []
        for index in range(args.repeats):
            start = time.perf_counter()
            output = pipe.generate(args.prompt, max_new_tokens=args.max_new_tokens)
            elapsed = time.perf_counter() - start
            text = str(output)
            count = token_count(pipe, text)
            timings.append(elapsed)
            counts.append(count)
            print(f"run {index + 1}/{args.repeats}: {count} tokens in {elapsed:.3f}s = {count / elapsed:.2f} tok/s")
        rates = [n / t for n, t in zip(counts, timings) if t > 0]
        record.update({
            "status": "success", "generated_tokens": counts, "elapsed_seconds": timings,
            "tokens_per_second": rates, "median_tokens_per_second": statistics.median(rates),
            "min_tokens_per_second": min(rates), "max_tokens_per_second": max(rates),
        })
    except Exception as exc:
        record.update({"status": "error", "exception": repr(exc), "traceback": traceback.format_exc()})
        print(record["traceback"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0 if record["status"] == "success" else 2

if __name__ == "__main__":
    raise SystemExit(main())
