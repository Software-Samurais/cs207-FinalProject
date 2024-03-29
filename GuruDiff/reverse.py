'''
with reference to: course CSE 599W Syetems for ML and framework of their homwwork (w/o solution)
1. https://dlsys.cs.washington.edu/
2. https://github.com/dlsys-course/assignment1
'''

import numpy as np

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.
            
            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object, 
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __sub__(self, other):
        if isinstance(other, Node):
            new_node = add_op(self, -other)
        else:
            new_node = add_byconst_op(self, -other)
        return new_node

    def __rsub__(self, other):
        if isinstance(other, Node):
            new_node = -add_op(self, -other)
        else:
            new_node = -add_byconst_op(self, -other)
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other**(-1))
        else: 
            new_node = mul_byconst_op(self, 1./other)
        return new_node
    
    def __rtruediv__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other**-1)**-1
        else: 
            new_node = mul_byconst_op(self, 1./other)**-1
        return new_node


    def __pow__(self, other): 
        assert not isinstance(other, Node)
        return power_op(self, other)

    def __neg__(self):
        return mul_byconst_op(self, -1)

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node

    def __eq__(self, other):
        if isinstance(other, Node):
            new_node = eq_op(self, other)
        else:
            new_node = eq_op(self, other*oneslike_op(self))
        return new_node
    def __ne__(self, other):
        return (self==other)==0

    def __hash__(self):
        return id(self)


    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name.""" 
        return self.name

    __repr__ = __str__

def Variable(name):
    """User defined variables in an expression.  
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node

class Op(object):
    """Op represents operations performed on nodes.
    Functions
    ---------
    __call__: return a Node which is created by this op
    compute: the only function which will return real values which will be used
             for computation of both forward and backward
    gradient: return a Node for the gradient computation for the reverse mode
    """
    def __call__(self):
        """Create a new node and associate the op object with the node.
        
        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

class AddOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [output_grad, output_grad]

class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad]

class EqOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s==%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return abs(input_vals[0] - input_vals[1])<1e-7

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        raise NotImplementedError




class MulOp(Op):
    """Op to element-wise multiply two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        return [output_grad*node.inputs[1], output_grad*node.inputs[0]]

class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""

        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of multiplication node, return gradient contribution to input."""

        return [output_grad * node.const_attr]

class MatMulOp(Op):
    """Op to matrix multiply two nodes."""
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""

        assert len(input_vals) == 2
        inp_A = input_vals[0].transpose() if node.matmul_attr_trans_A else input_vals[0]
        inp_B = input_vals[1].transpose() if node.matmul_attr_trans_B else input_vals[1]
        return np.matmul(inp_A, inp_B)

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.
            
        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """

        return [matmul_op(output_grad   , node.inputs[1], False, True), \
                matmul_op(node.inputs[0], output_grad   , True , False)]


class SinOp(Op):
    """Op to element-wise sin of node."""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "sin(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sin(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * cos_op(node.inputs[0])]

class CosOp(Op):
    """Op to element-wise cos of node."""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "cos(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.cos(input_vals[0])

    def gradient(self, node, output_grad):
        return [-sin_op(node.inputs[0]) * output_grad]

class TanOp(Op):
    """Op to element-wise tan of node."""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "tan(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sin(input_vals[0])/np.cos(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad*(cos_op(node.inputs[0])**-2)]


class SinhOp(Op):
    """Op to element-wise sinh of node."""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "sinh(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sinh(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * cosh_op(node.inputs[0])]

class CoshOp(Op):
    """Op to element-wise cosh of node."""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "cosh(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.cosh(input_vals[0])

    def gradient(self, node, output_grad):
        return [sinh_op(node.inputs[0]) * output_grad]

class TanhOp(Op):
    """Op to element-wise tanh of node."""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "tanh(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.tanh(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad*(
            (cosh_op(node.inputs[0])**2 - sinh_op(node.inputs[0])**2) 
            * (cosh_op(node.inputs[0])**-2)
        )]


class ArcSinOp(Op):

    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "arcsin(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert (input_vals[0]>-1).all() and (input_vals[0]<1).all(), 'input of arcsin must in (-1,1)'
        return np.arcsin(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * ((1-node.inputs[0]**2)**-0.5)]

class ArcCosOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "arccos(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert (input_vals[0]>-1).all() and (input_vals[0]<1).all(), 'input of arccos must in (-1,1)'
        return np.arccos(input_vals[0])

    def gradient(self, node, output_grad):
        return [-output_grad * ((1-node.inputs[0]**2)**-0.5)]

class ArcTanOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "arctan(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.arctan(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad/(1 + node.inputs[0]**2)]

class PowerOp(Op):
    """Op to element-wise do power of a node."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "%s**%s" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0]**node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad * \
                (node.const_attr * (node.inputs[0]**(node.const_attr-1)))]

class ExpOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "%s**%s" % (str(const_val), node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return node.const_attr**input_vals[0]

    def gradient(self, node, output_grad):
        return [output_grad * \
                (np.log(node.const_attr) * \
                exp_op(node.inputs[0], node.const_attr))]

class LogOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "log(%s,%s)" % (str(const_val), node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.log(input_vals[0]) / np.log(node.const_attr)

    def gradient(self, node, output_grad):
        return [output_grad * \
                ((np.log(node.const_attr) * node.inputs[0])**-1)]


class LogisticOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "logistic(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return 1./(1+np.exp(-input_vals[0]))

    def gradient(self, node, output_grad):
        return [output_grad*\
                (1./(1+exp_op(-node.inputs[0], np.e)))*(1-1./(1+exp_op(-node.inputs[0], np.e)))]

class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None

class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        '''
        Parameters
        ----------
        node: node that performs the gradient
        output_grad: gradient of this node, return of this function will be passed
                     to its inputs. It is worth noticing that the return of this function
                     is a list of Node. 
        '''
        return [zeroslike_op(node.inputs[0])]

class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
cos_op = CosOp()
sin_op = SinOp()
tan_op = TanOp()
cosh_op = CoshOp()
sinh_op = SinhOp()
tanh_op = TanhOp()
arccos_op = ArcCosOp()
arcsin_op = ArcSinOp()
arctan_op = ArcTanOp()
power_op = PowerOp()
exp_op = ExpOp()
log_op = LogOp()
logistic_op = LogisticOp()
eq_op = EqOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()

class Executor:
    """Executor computes values for a given subset of nodes in a computation graph.""" 
    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        in this function, we only have to deal with the compute function of each node's op
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list. 
        """
        node_to_val_map = dict(feed_dict)
        # print('init val_map:', node_to_val_map)
        # Traverse graph in topological sort order and compute values for all nodes.
        topo_order = find_topo_sort(self.eval_node_list)
        # print('topo_order:', list(topo_order))
        for _i, node in enumerate(topo_order):
            # print('-'*30)
            # print(_i, node)
            if isinstance(node.op, PlaceholderOp):
                continue
            # print('inputs:', _i, node.inputs)
            # print('inputs map:', _i, [node_to_val_map[i] for i in node.inputs])
            node_to_val_map[node] = node.op.compute(node, 
                                    [node_to_val_map[i] for i in node.inputs])

        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results

def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_node])))

    for _idx, node in enumerate(reverse_topo_order):
        # print('-'*30)
        # print(_idx, node)
        # print('op', node.op)
        # print('grad list', node_to_output_grads_list[node])
        grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = grad
        input_grads = node.op.gradient(node, grad)
        
        # print('grad:', grad)
        # print('inp_grads:', input_grads)
        # print('inp:', node.inputs)
        for _i, inp_node in enumerate(node.inputs):
            node_to_output_grads_list[inp_node] = \
            node_to_output_grads_list[inp_node] + [input_grads[_i]] \
            if inp_node in node_to_output_grads_list.keys()\
            else \
            [input_grads[_i]]

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##############################
####### Helper Methods ####### 
##############################

def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    
    A simple algorithm is to do a post-order DFS traversal on the given nodes, 
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
