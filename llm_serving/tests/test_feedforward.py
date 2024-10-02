import os
import pytest

import torch

from model import FeedForward


class Config:
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size


@pytest.fixture
def config():
    # Create a simple configuration object
    return Config(hidden_size=64, intermediate_size=256)


@pytest.fixture
def model(config):
    # Initialize the FeedForward model
    return FeedForward(config)


def test_feedforward_shape(model):
    # Create a random input tensor with the shape [batch_size, seq_len, hidden_size]
    batch_size, seq_len = 2, 128
    hidden_size = model.dense_h_to_4h.in_features

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)

    # Perform the forward pass
    output_tensor = model(input_tensor)

    # Check if the output shape is correct
    assert output_tensor.shape == (batch_size, seq_len, hidden_size)

    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'input_tensor': input_tensor,
    #     'output_tensor': output_tensor,
    # }, 'test_feedforward_data.pth')


def test_feedforward_value(model):
    path = os.path.join(os.path.dirname(__file__), 'test_feedforward_data.pth')
    loaded = torch.load(path, weights_only=True)
    model.load_state_dict(loaded['model_state_dict'])
    input_tensor = loaded['input_tensor']
    reference_output_tensor = loaded['output_tensor']

    # Perform the forward pass
    output_tensor = model(input_tensor)
    torch.testing.assert_close(
        output_tensor, reference_output_tensor, rtol=1e-5, atol=1e-5)
