import pytest
import torch
from model import KVCache


@pytest.fixture
def kv_cache():
    return KVCache()


def test_kv_cache_initialization(kv_cache):
    # Test if the key and value are initialized as None
    assert kv_cache.key is None
    assert kv_cache.value is None


def test_kv_cache_update_initial(kv_cache):
    # Test the update method when cache is empty
    # Example tensor: [batch_size, num_attention_heads, seq_len, head_size]
    key = torch.randn(2, 4, 8, 16)
    value = torch.randn(2, 4, 8, 16)

    updated_key, updated_value = kv_cache.update(key, value)

    # Check if the key and value are updated correctly
    assert torch.equal(updated_key, key)
    assert torch.equal(updated_value, value)
    assert kv_cache.key is not None
    assert kv_cache.value is not None


def test_kv_cache_update_concat(kv_cache):
    # Test the update method when cache already has key and value
    key1 = torch.randn(2, 4, 8, 16)
    value1 = torch.randn(2, 4, 8, 16)
    kv_cache.update(key1, value1)

    # Update with new keys and values
    key2 = torch.randn(2, 4, 4, 16)
    value2 = torch.randn(2, 4, 4, 16)
    updated_key, updated_value = kv_cache.update(key2, value2)

    # Expected concatenated sizes
    assert updated_key.size(-2) == key1.size(-2) + key2.size(-2)
    assert updated_value.size(-2) == value1.size(-2) + value2.size(-2)

    # Check if the key and value are updated correctly
    assert torch.equal(updated_key[..., :8, :], key1)
    assert torch.equal(updated_key[..., 8:, :], key2)
    assert torch.equal(updated_value[..., :8, :], value1)
    assert torch.equal(updated_value[..., 8:, :], value2)


def test_past_key_values_length(kv_cache):
    # Test the past_key_values_length method
    assert kv_cache.past_key_values_length() == 0  # Initially, should be 0

    key = torch.randn(2, 4, 8, 16)
    value = torch.randn(2, 4, 8, 16)
    kv_cache.update(key, value)

    # After update, should be the seq_len
    assert kv_cache.past_key_values_length() == 8
