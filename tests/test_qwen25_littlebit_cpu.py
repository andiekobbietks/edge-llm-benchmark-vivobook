from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from quantization.utils.quant_util import apply_littlebit_patch


def make_tiny_qwen25_littlebit_model():
    """Build a small Qwen2.5-family graph and replace its Linear layers with LittleBit."""
    torch.manual_seed(11)
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_config(config)
    args = SimpleNamespace(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        quant_func="STEBinary",
        quant_mod="LittleBitLinear",
        residual=False,
        split_dim=8,
        eff_bit=None,
        min_split_dim=8,
        kv_factor=1.0,
        use_itq=False,
        itq_n_iter=5,
    )
    model = apply_littlebit_patch(model, args, do_train=True)
    return model.eval()


def test_qwen25_architecture_runs_littlebit_inference_on_cpu():
    model = make_tiny_qwen25_littlebit_model()
    input_ids = torch.randint(0, 256, (1, 8), dtype=torch.long)

    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False)

    assert output.logits.shape == (1, 8, 256)
    assert torch.isfinite(output.logits).all()
