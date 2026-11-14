"""End-to-end CPU-only LittleBit checkpoint smoke test.

This deliberately uses the public Qwen2.5-0.5B-Instruct base model, applies
LittleBit's SVD factorisation, saves a local packed checkpoint, reloads it
through LittleBitModel.from_pretrained(device="cpu"), and generates tokens.

It does NOT perform quantisation-aware training/distillation. The purpose is
to validate full-model CPU checkpoint conversion/loading/generation, not model
quality or equivalence to a trained LittleBit release.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization.hub import LittleBitConfig, LittleBitModel
from quantization.modules import LittleBitLinear
from quantization.utils.quant_util import apply_littlebit_patch


BASE_MODEL = os.environ.get("LITTLEBIT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUTPUT_DIR = Path(os.environ.get("LITTLEBIT_OUTPUT_DIR", "artifacts/qwen25-05b-littlebit-cpu"))


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    started = time.perf_counter()
    print(f"Loading base model on CPU: {BASE_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
    model.eval()

    quant_args = SimpleNamespace(
        model_id=BASE_MODEL,
        quant_func="STEBinary",
        quant_mod="LittleBitLinear",
        residual=False,
        split_dim=256,
        eff_bit=0.55,
        min_split_dim=8,
        kv_factor=1.0,
        use_itq=False,
        itq_n_iter=5,
    )
    print("Applying LittleBit factorisation on CPU (SVD initialisation; no QAT).", flush=True)
    model = apply_littlebit_patch(model, quant_args, do_train=True)
    model.eval()
    assert any(isinstance(m, LittleBitLinear) for m in model.modules())
    assert all(p.device.type == "cpu" for p in model.parameters())

    config = LittleBitConfig(
        quant_func="STEBinary",
        eff_bit=0.55,
        split_dim=256,
        residual=False,
        kv_factor=1.0,
        min_split_dim=8,
        use_itq=False,
        itq_n_iter=5,
    )
    wrapped = LittleBitModel(model=model, config=config, base_model_id=BASE_MODEL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wrapped._save_pretrained(OUTPUT_DIR)
    print(f"Saved packed LittleBit checkpoint: {OUTPUT_DIR}", flush=True)

    del wrapped, model
    import gc
    gc.collect()

    print("Reloading packed checkpoint explicitly onto CPU.", flush=True)
    reloaded = LittleBitModel.from_pretrained(
        str(OUTPUT_DIR),
        torch_dtype=torch.float32,
        device="cpu",
    )
    reloaded.eval()
    assert all(p.device.type == "cpu" for p in reloaded.parameters())

    prompt = "Explain why running language models on a CPU can be useful."
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        generated = reloaded.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            use_cache=True,
        )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    assert generated.shape[1] > inputs["input_ids"].shape[1]
    assert decoded.strip()
    elapsed = time.perf_counter() - started

    report = {
        "base_model": BASE_MODEL,
        "method": "LittleBit SVD factorisation + STEBinary packing",
        "quantisation_aware_training": False,
        "device": "CPU",
        "cuda_available": torch.cuda.is_available(),
        "generated_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
        "elapsed_seconds_including_download_and_conversion": round(elapsed, 3),
        "output_text": decoded,
        "status": "PASS",
        "quality_claim": "none; this is an untrained factorisation smoke test",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "cpu-smoke-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
