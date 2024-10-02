import os
import pytest

import torch

from model import Attention


class Config:
    def __init__(self, hidden_size, num_attention_heads):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rotary_pct = 0.25
        self.rotary_emb_base = 10000
        self.max_position_embeddings = 2048
        self.attention_bias = True
        self.attention_dropout = 0.0


@pytest.fixture
def config():
    # Create a simple configuration object
    return Config(hidden_size=128, num_attention_heads=4)


@pytest.fixture
def model(config):
    # Initialize the FeedForward model
    return Attention(config)


def test_attention_shape(model):
    # Test the shape of the attention output
    batch_size = 2
    seq_len = 8
    hidden_size = 128
    num_attention_heads = 4
    attn_head_size = hidden_size // num_attention_heads

    query_tensor = torch.randn(batch_size, num_attention_heads, seq_len, attn_head_size)
    key_tensor = torch.randn(batch_size, num_attention_heads, seq_len, attn_head_size)
    value_tensor = torch.randn(batch_size, num_attention_heads, seq_len, attn_head_size)

    attention_output = model._attn(query_tensor, key_tensor, value_tensor)

    # Check if the output shape is correct
    assert attention_output.size() == (batch_size, seq_len, hidden_size)

    # torch.save({
    #     "query_tensor": query_tensor,
    #     "key_tensor": key_tensor,
    #     "value_tensor": value_tensor,
    #     "attention_output": attention_output
    # }, "attention_output.pth")


def test_attention_value(model):
    path = os.path.join(os.path.dirname(__file__), "attention_output.pth")
    loaded = torch.load(path, weights_only=True)
    query_tensor = loaded["query_tensor"]
    key_tensor = loaded["key_tensor"]
    value_tensor = loaded["value_tensor"]
    reference_attention_output = loaded["attention_output"]

    # Perform the forward pass
    attention_output = model._attn(query_tensor, key_tensor, value_tensor)
    torch.testing.assert_close(
        attention_output, reference_attention_output, rtol=1e-5, atol=1e-5)
