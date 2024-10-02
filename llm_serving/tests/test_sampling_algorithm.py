import pytest
import torch

from generate import top_k_, top_p_


def test_top_k_():
    # Generate random logits
    logits = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
    
    # Set top_k=3, meaning only the top 3 logits should remain
    filtered_logits = top_k_(logits, top_k=3)

    # Check that the top 3 values are kept and others are filtered
    assert filtered_logits[0, 0] == -float("Inf")  # The first value should be filtered out
    assert filtered_logits[0, 1] == -float("Inf")  # The second value should be filtered out
    assert filtered_logits[0, 2] == 30.0  # The third value should be kept
    assert filtered_logits[0, 3] == 40.0  # The fourth value should be kept
    assert filtered_logits[0, 4] == 50.0  # The fifth value should be kept


def test_top_p_():
    # Generate logits
    logits = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
    
    # Set top_p=0.7, meaning the logits with cumulative probability above 0.7 should be filtered
    filtered_logits = top_p_(logits, top_p=0.7)

    # Check that the logits have been filtered correctly
    assert filtered_logits[0, 4] == 50.0  # The highest value should be filtered out
    assert filtered_logits[0, 3] == -float("Inf")  # This value should remain
    assert filtered_logits[0, 2] == -float("Inf")  # This value should remain
    assert filtered_logits[0, 1] == -float("Inf")  # This value should remain
    assert filtered_logits[0, 0] == -float("Inf")  # This value should remain

    # test min_tokens_to_keep
    filtered_logits = top_p_(logits, top_p=0.7, min_tokens_to_keep=2)
    assert filtered_logits[0, 4] == 50.0  # The highest value should be filtered out
    assert filtered_logits[0, 3] == 40.0  # This value should remain
    assert filtered_logits[0, 2] == -float("Inf")  # This value should remain
    assert filtered_logits[0, 1] == -float("Inf")  # This value should remain
    assert filtered_logits[0, 0] == -float("Inf")  # This value should remain
