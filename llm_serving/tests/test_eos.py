import os
import pytest

import torch
from transformers import AutoTokenizer

from generate import stop_on_eos


@pytest.fixture
def tokenizer():
    # Create a simple configuration object
    orgize_dir = os.path.join(os.path.dirname(__file__), "../checkpoints/EleutherAI/")
    model_name = os.listdir(orgize_dir)[0]
    tokenizer = AutoTokenizer.from_pretrained(orgize_dir + model_name, clean_up_tokenization_spaces=True)
    return tokenizer


def test_stop_on_eos_true(tokenizer):
    # Test that stop_on_eos returns True for the EOS token
    eos_token = torch.tensor(tokenizer.eos_token_id)  # Use tokenizer to get EOS token ID
    assert stop_on_eos(eos_token) is True


def test_stop_on_eos_false(tokenizer):
    # Test that stop_on_eos returns False for a non-EOS token
    non_eos_token = torch.tensor(tokenizer.encode("Hello"))  # Encode a regular token
    assert stop_on_eos(non_eos_token) is False
