import os
import platform
import pytest
import warnings

import torch

from transformers import AutoTokenizer
from model import PythiaForCausalLM

from generate import prefill, decode
from model import KVCache


@pytest.fixture
def tokerizer_and_model():
    precision = torch.bfloat16
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "../checkpoints/EleutherAI/pythia-410m")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, clean_up_tokenization_spaces=True)
    model = PythiaForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=precision)
    return tokenizer, model


def test_prefill(tokerizer_and_model):
    tokenizer, model = tokerizer_and_model
    input_text = "My name is "
    encoded = tokenizer(input_text, return_tensors="pt")
    generator = torch.Generator().manual_seed(6216)
    next_token, kvcaches = prefill(model, encoded["input_ids"], encoded['attention_mask'], generator=generator)

    assert len(next_token.size()) == 0
    assert len(kvcaches) == model.config.num_hidden_layers and all([isinstance(kvcache, KVCache) for kvcache in kvcaches])

    if platform.system() == 'Darwin':
        assert next_token.item() == 2015
    elif platform.system() in ['Linux', 'Windows']:
        assert next_token.item() == 187
    else:
        warnings.warn(f"Unknown platform: {platform.system()}")


def test_decode(tokerizer_and_model):
    tokenizer, model = tokerizer_and_model
    input_text = "My name is "
    encoded = tokenizer(input_text, return_tensors="pt")
    generator = torch.Generator().manual_seed(6216)
    next_token, kvcaches = prefill(model, encoded["input_ids"], encoded['attention_mask'], generator=generator)

    assert len(next_token.size()) == 0
    assert len(kvcaches) == model.config.num_hidden_layers and all([isinstance(kvcache, KVCache) for kvcache in kvcaches])

    if platform.system() == 'Darwin':
        assert next_token.item() == 2015
    elif platform.system() in ['Linux', 'Windows']:
        assert next_token.item() == 187
    else:
        warnings.warn(f"Unknown platform: {platform.system()}")

    generated_tokens = decode(model, kvcaches, next_token.view(
        1, -1), encoded['attention_mask'], num_new_tokens=1, generator=generator)

    assert len(generated_tokens) == 1

    if platform.system() == 'Darwin':
        assert generated_tokens[0].item() == 1134
    elif platform.system() == 'Linux':
        assert generated_tokens[0].item() in [9, 40917]
    elif platform.system() == 'Windows':
        assert generated_tokens[0].item() == 6275
    else:
        warnings.warn(f"Unknown platform: {platform.system()}")
