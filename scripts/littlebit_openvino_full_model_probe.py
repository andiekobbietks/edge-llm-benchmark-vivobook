"""Full-model LittleBit OpenVINO CPU validation and no-KV-cache greedy decode.

This proves graph-level autoregressive token generation by recomputing the full
prefix each step. It is a correctness/feasibility probe, not a speed claim.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import openvino as ov
import torch
from openvino import PartialShape
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization.utils.quant_util import apply_littlebit_patch

BASE_MODEL = os.environ.get("LITTLEBIT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("LITTLEBIT_OV_MAX_NEW_TOKENS", "8"))

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
    prompt = tokenizer("Explain why local language model inference can be useful.",
                       return_tensors="pt")
    wrapper = LogitsOnly(model).eval()
    with torch.inference_mode():
        expected = wrapper(prompt["input_ids"], prompt["attention_mask"]).cpu().numpy()
    print("Converting complete LittleBit model to OpenVINO CPU.", flush=True)
    ov_model = ov.convert_model(wrapper,
        example_input=(prompt["input_ids"], prompt["attention_mask"]))
    ov_model.reshape({
        ov_model.input(0): PartialShape([1, -1]),
        ov_model.input(1): PartialShape([1, -1]),
    })
    compiled = ov.Core().compile_model(ov_model, "CPU")
    inputs = prompt["input_ids"].numpy().astype(np.int64)
    mask = prompt["attention_mask"].numpy().astype(np.int64)
    logits = compiled({"input_ids": inputs, "attention_mask": mask})[compiled.output(0)]
    max_abs = float(np.max(np.abs(logits - expected)))
    denom = float(np.linalg.norm(expected) * np.linalg.norm(logits))
    cosine = float(np.sum(expected * logits) / denom) if denom else 0.0
    generated = []
    for step in range(MAX_NEW_TOKENS):
        next_id = int(np.argmax(logits[0, -1, :]))
        generated.append(next_id)
        inputs = np.concatenate([inputs, np.array([[next_id]], dtype=np.int64)], axis=1)
        mask = np.ones_like(inputs, dtype=np.int64)
        logits = compiled({"input_ids": inputs, "attention_mask": mask})[compiled.output(0)]
        print(f"OpenVINO greedy token {step+1}/{MAX_NEW_TOKENS}: {next_id}", flush=True)
    text_out = tokenizer.decode(generated, skip_special_tokens=True)
    report = {
        "base_model": BASE_MODEL, "device": "OpenVINO CPU",
        "openvino_version": ov.__version__,
        "graph": "full LittleBit Qwen2.5-0.5B forward graph with dynamic sequence axis",
        "max_absolute_error_vs_pytorch": max_abs, "cosine_similarity_vs_pytorch": cosine,
        "greedy_decode_tokens": len(generated), "generated_token_ids": generated,
        "generated_text": text_out, "kv_cache": False,
        "decode_method": "autoregressive greedy; recompute entire prefix each token",
        "elapsed_seconds": round(time.perf_counter()-started, 3),
        "status": "PASS" if np.isfinite(logits).all() and max_abs <= 5e-3 and len(generated)==MAX_NEW_TOKENS else "FAIL",
        "limitations": ["No KV cache; not a throughput benchmark",
                        "SVD initialisation only; no QAT or distillation",
                        "This is hosted x86 CPU, not Intel Iris Xe validation"]
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/littlebit-openvino-full-model-probe.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
