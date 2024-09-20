from typing import Any, Dict, List

import numpy as np


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[np.ndarray]
            The input values of the given node.

        Returns
        -------
        output: np.ndarray
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of input values."""
        """TODO: Your code here"""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        """TODO: Your code here"""
        inp_1, inp_2 = node.inputs[0], node.inputs[1]
        return [mul(output_grad, inp_2), mul(output_grad, inp_1)]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of the input value and the constant."""
        """TODO: Your code here"""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        """TODO: Your code here"""
        return [output_grad * node.constant]


class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of input values."""
        """TODO: Your code here"""
        assert len(input_values) == 2
        return input_values[0] / input_values[1]


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        """TODO: Your code here"""
        x1 = node.inputs[0]
        x2 = node.inputs[1]
        return [output_grad/x2, (output_grad * (-1) * x1) / (x2 * x2)]


class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of the input value and the constant."""
        """TODO: Your code here"""
        assert len(input_values) == 1
        return input_values[0] / node.constant


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        """TODO: Your code here"""
        return [output_grad / node.constant]


class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node, trans_A: bool = False, trans_B: bool = False
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix
        trans_A: bool
            A boolean flag denoting whether to transpose A before multiplication.
        trans_B: bool
            A boolean flag denoting whether to transpose B before multiplication.

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"trans_A": trans_A, "trans_B": trans_B},
            name=f"({node_A.name + ('.T' if trans_A else '')}@{node_B.name + ('.T' if trans_B else '')})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the matrix multiplication result of input values.

        Note
        ----
        For this homework, you can assume the matmul only works for 2d matrices.
        That being said, the test cases guarantee that input values are
        always 2d numpy.ndarray.
        """
        """TODO: Your code here"""
        assert len(input_values) == 2
        if node.trans_A:
            input_values[0] = input_values[0].T
        if node.trans_B:
            input_values[1] = input_values[1].T
        assert input_values[0].shape[1] == input_values[1].shape[0]
        return input_values[0]@input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input.

        Note
        ----
        - Same as the `compute` method, you can assume that the input are 2d matrices.
        However, it would be a good exercise to think about how to handle
        more general cases, i.e., when input can be either 1d vectors,
        2d matrices, or multi-dim tensors.
        - You may want to look up some materials for the gradients of matmul.
        """
        """TODO: Your code here"""
        x1 = node.inputs[0]
        x2 = node.inputs[1]
        trans_b = not(node.trans_B)
        trans_a = not(node.trans_A)
        x1_grad = matmul(output_grad, x2, trans_B=trans_b)
        x2_grad = matmul(x1, output_grad, trans_A=trans_a)
        return [x1_grad, x2_grad]


class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.zeros(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.ones(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class BroadCastOp(Op):

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "original_shape": None,
                "target_shape": None,
            },
            name=f"({node_A.name}).broadcast_to({node_B.name})",
        )
    
    def compute(self, node: Node, input_values: List[np.ndarray]):
        assert len(input_values) == 2
        x = input_values[0]
        shape = input_values[1].shape
        y = np.broadcast_to(x, shape)
        return y

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        zero_grad = zeros_like(node.inputs[1])
        reduced_grad = sum_op(output_grad, axis=0, keepdims=False)
        return [reduced_grad, zero_grad]
    



class SumOp(Op):
    """Op to perform summation over specified axes."""

    def __call__(self, node_A: Node, axis=None, keepdims=False) -> Node:
        """Create a sum operation node."""
        return Node(
            inputs=[node_A],
            op=self,
            attrs={
                "axis": axis, 
                "keepdims": keepdims,
                "original_shape": None,
            },
            name=f"sum({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Compute the summation over specified axes."""
        assert len(input_values) == 1
        node.original_shape = input_values[0].shape
        axis = node.attrs["axis"]
        keepdims = node.attrs["keepdims"]
        return np.sum(input_values[0], axis=axis, keepdims=keepdims)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the summation, return the gradient of the input."""
        input_shape = node.original_shape
        axis = node.attrs["axis"]
        keepdims = node.attrs["keepdims"]

        # If keepdims is False, expand output_grad to match input shape using BroadcastToOp
        if not keepdims and axis is not None:
            expanded_output_grad = broadcast_to(output_grad, input_shape)
        else:
            expanded_output_grad = output_grad

        return [expanded_output_grad]

class ExpOp(Op):
    """Element-wise exponential operation with caching."""
    
    def __call__(self, node_A: Node) -> Node:
        """Create a new node that represents the exp operation."""
        return Node(
            inputs=[node_A],
            op=self,
            attrs={
                "exp_calc": None, 
            },
            name=f"exp({node_A.name})"
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Compute the exponential of input values and store the result in node.attrs."""
        assert len(input_values) == 1
        exp_x = np.exp(input_values[0])
        return exp_x

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Compute the gradient of the exp operation using cached value."""
        calc_exp_x = exp(node.inputs[0])
        return [mul(output_grad, calc_exp_x)]

class LogOp(Op):
    """Log operation with input caching."""
    
    def __call__(self, node_A: Node) -> Node:
        """Create a new node that represents the exp operation."""
        return Node(
            inputs=[node_A],
            op=self,
            name=f"log({node_A.name})",
            attrs={
                "input_val": None, 
            }
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Compute the exponential of input values and store the result in node.attrs."""
        assert len(input_values) == 1
        log_x = np.log(input_values[0])
        return log_x

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Compute the gradient of the exp operation using cached value."""
        input_val = node.inputs[0]
        return [div(output_grad, input_val)]


# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
broadcast = BroadCastOp()
sum_op = SumOp()
exp = ExpOp()
log = LogOp()

def topologicalSortUtil(node, visited, sorted_nodes):
    visited.add(node)

    for i in node.inputs:
        if i not in visited:
            topologicalSortUtil(i, visited, sorted_nodes)

    sorted_nodes.append(node)


class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes
        self.sorted_nodes = []


    def run(self, input_values: Dict[Node, np.ndarray]) -> List[np.ndarray]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, np.ndarray]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[np.ndarray]
            The list of values for nodes in `eval_nodes` field.
        """
        """TODO: Your code here"""
        visited = set()
        sorted_nodes = []
        for node in self.eval_nodes:
            topologicalSortUtil(node, visited, sorted_nodes)

        computed = {}
        for node in sorted_nodes:
            if len(node.inputs) == 0:
                input_values_for_node = input_values[node]
                computed[node] = input_values[node]
            else:
                input_nodes = node.inputs
                input_values_for_node = []
                for inp_node in input_nodes:
                    if inp_node in computed:
                        input_values_for_node.append(computed[inp_node])
                    else:
                        input_values_for_node.append(input_values[inp_node])
                computed[node] = node.op.compute(node, input_values_for_node)
        return [computed[node] for node in self.eval_nodes]




def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """

    """TODO: Your code here"""
    grad_node = Node(inputs=[], op=None, attrs={"constant": 1.0}, name="input_node")
    node_to_grad = {
        output_node: [ones_like(output_node)]
    }
    sorted_nodes = []
    visited = set()
    topologicalSortUtil(output_node, visited, sorted_nodes)
   
    for i in reversed(sorted_nodes):
        delta_v_i = node_to_grad[i][0]
        for grad in node_to_grad[i][1:]:
            delta_v_i = add(delta_v_i, grad)
        for n, k in enumerate(i.inputs):
            delta_v_k_i = i.op.gradient(i, delta_v_i)[n]
            if k not in node_to_grad:
                node_to_grad[k] = []
            node_to_grad[k].append(delta_v_k_i)
    grad_array = []
    for node in nodes:
        if node in node_to_grad:
            adj = node_to_grad[node][0]
            for grad in node_to_grad[node][1:]:
                adj = add(adj, grad)
        grad_array.append(adj)

    return grad_array