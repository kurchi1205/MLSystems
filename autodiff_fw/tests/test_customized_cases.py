"""We encourage you to create your own test cases, which helps
you confirm the correctness of your implementation.

If you are interested, you can write your own tests in this file
and share them with us by including this file in your submission.
Please make the tests "pytest compatible" by starting each test
function name with prefix "test_".

We appreciate it if you can share your tests, which can help
improve this course and the homework. However, please note that
this part is voluntary -- you will not get more scores by sharing
test cases, and conversely, will not get fewer scores if you do
not share.
"""

from typing import Dict, List
import numpy as np
import sys
sys.path.insert(0, "./")
import auto_diff as ad

def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, np.ndarray],
    expected_outputs: List[np.ndarray],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        np.testing.assert_allclose(actual=output_val, desired=expected_val)

def test_log_operation():
    x = ad.Variable("x")
    y = ad.log(x)
    y_grad = ad.gradients(y, nodes=[x])[0]
    evaluator = ad.Evaluator(eval_nodes=[y, y_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x: np.array([[2.0, 3.0], [4.0, 5.0]]),
        },
        expected_outputs=[
            np.log(np.array([[2.0, 3.0], [4.0, 5.0]])),   # Expected output for log
            1 / np.array([[2.0, 3.0], [4.0, 5.0]]),       # Expected gradient of log
        ],
    )

def test_exp_operation():
    x = ad.Variable("x")
    y = ad.exp(x)
    y_grad = ad.gradients(y, nodes=[x])[0]
    evaluator = ad.Evaluator(eval_nodes=[y, y_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x: np.array([[2.0, 3.0], [4.0, 5.0]]),
        },
        expected_outputs=[
            np.exp(np.array([[2.0, 3.0], [4.0, 5.0]])),
            np.exp(np.array([[2.0, 3.0], [4.0, 5.0]])),
        ],
    )



if __name__ == "__main__":
    test_log_operation()
    test_exp_operation()