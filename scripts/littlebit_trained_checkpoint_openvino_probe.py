"""Validate a trained LittleBit checkpoint on OpenVINO CPU.

Unlike the SVD-only probe, this script loads an actual QAT checkpoint produced
by Samsung's trainer. It checks PyTorch/OpenVINO logits, greedy generation,
and pathological repetition. It is a correctness/quality gate, not a speed claim.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from openvino import PartialShape
from torch import nn
from transformers import AutoTokenizer
from quantization.hub import LittleBitModel

CHECKPOINT = os.environ["LITTLEBIT_CHECKPOINT"]
MAX_NEW_TOKENS = int(os.environ.get("LITTLEBIT_OV_MAX_NEW_TOKENS", "32"))


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    started = time.perf_counter()
    print(f"Loading trained LittleBit checkpoint: {CHECKPOINT}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = LittleBitModel.from_pretrained(
        CHECKPOINT, torch_dtype=torch.float32, device="cpu"
    )
    model.eval()

    prompt = tokenizer(
        "Explain why local language model inference can be useful.",
        return_tensors="pt",
    )
    wrapper = LogitsOnly(model).eval()
    with torch.inference_mode():
        expected = wrapper(prompt["input_ids"], prompt["attention_mask"]).cpu().numpy()

    print("Converting trained LittleBit checkpoint to OpenVINO CPU.", flush=True)
    ov_model = ov.convert_model(
        wrapper,
        example_input=(prompt["input_ids"], prompt["attention_mask"]),
    )
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
        print(f"trained OpenVINO greedy token {step + 1}/{MAX_NEW_TOKENS}: {next_id}", flush=True)

    unique_count = len(set(generated))
    longest_run = 1
    current_run = 1
    for previous, current in zip(generated, generated[1:]):
        current_run = current_run + 1 if current == previous else 1
        longest_run = max(longest_run, current_run)
    repetition = (
        len(generated) >= 8
        and (unique_count <= 3 or longest_run >= 6)
    )
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    numerical_pass = bool(
        np.isfinite(logits).all()
        and np.isfinite(max_abs)
        and max_abs <= 5e-3
        and len(generated) == MAX_NEW_TOKENS
    )
    report = {
        "checkpoint": CHECKPOINT,
        "device": "OpenVINO CPU",
        "openvino_version": ov.__version__,
        "max_absolute_error_vs_pytorch": max_abs,
        "cosine_similarity_vs_pytorch": cosine,
        "greedy_decode_tokens": len(generated),
        "generated_token_ids": generated,
        "generated_text": decoded,
        "unique_generated_tokens": unique_count,
        "longest_same_token_run": longest_run,
        "repetition_detected": repetition,
        "numerical_status": "PASS" if numerical_pass else "FAIL",
        "quality_status": "FAIL_REPETITION" if repetition else "REQUIRES_HUMAN_REVIEW",
        "status": "PASS" if numerical_pass and not repetition else "FAIL",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "No KV cache; not a throughput benchmark",
            "Greedy repetition heuristic is a screening gate, not a semantic quality metric",
            "Run held-out perplexity and human task review before claiming quality recovery",
        ],
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/littlebit-trained-openvino-quality-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
