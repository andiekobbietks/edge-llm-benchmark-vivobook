"""Probe conversion of the complete factorised Qwen model to OpenVINO CPU.

This tests a full-model forward/logits graph, not autoregressive GenAI export.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openvino as ov
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization.utils.quant_util import apply_littlebit_patch

BASE_MODEL = os.environ.get("LITTLEBIT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=False)[0]

def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        device_map={"": "cpu"})
    model.eval()
    args = SimpleNamespace(model_id=BASE_MODEL, quant_func="STEBinary",
        quant_mod="LittleBitLinear", residual=False, split_dim=256,
        eff_bit=0.55, min_split_dim=8, kv_factor=1.0, use_itq=False, itq_n_iter=5)
    model = apply_littlebit_patch(model, args, do_train=True)
    model.eval()
    encoded = tokenizer("Explain why local language model inference can be useful.",
                        return_tensors="pt")
    wrapper = LogitsOnly(model).eval()
    with torch.inference_mode():
        expected = wrapper(encoded["input_ids"], encoded["attention_mask"]).cpu().numpy()
    print("Converting full LittleBit Qwen forward graph to OpenVINO.", flush=True)
    ov_model = ov.convert_model(wrapper,
        example_input=(encoded["input_ids"], encoded["attention_mask"]))
    compiled = ov.Core().compile_model(ov_model, "CPU")
    actual = compiled({"input_ids": encoded["input_ids"].numpy(),
                       "attention_mask": encoded["attention_mask"].numpy()})[compiled.output(0)]
    max_abs = float(np.max(np.abs(actual - expected)))
    denom = float(np.linalg.norm(expected) * np.linalg.norm(actual))
    cosine = float(np.sum(expected * actual) / denom) if denom else 0.0
    report = {
        "base_model": BASE_MODEL,
        "graph": "full LittleBit-factorised Qwen2.5-0.5B forward pass; use_cache=False",
        "device": "OpenVINO CPU", "openvino_version": ov.__version__,
        "logits_shape": list(actual.shape), "max_absolute_error": max_abs,
        "cosine_similarity": cosine, "elapsed_seconds": round(time.perf_counter()-started, 3),
        "status": "PASS" if np.isfinite(actual).all() and max_abs <= 5e-3 else "FAIL",
        "scope": "full-model forward graph only; autoregressive generation/export not established"
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/littlebit-openvino-full-model-probe.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
