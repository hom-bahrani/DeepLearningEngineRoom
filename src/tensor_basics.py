import numpy as np
import matplotlib.pyplot as plt

# We'll implement our own basic tensor operations to understand the fundamentals
# In a real project, you would use PyTorch or TensorFlow instead

class Tensor:
    """
    A simple tensor implementation to understand the basics of tensors.
    This is a teaching implementation - use PyTorch or TensorFlow for real work.
    """
    
    def __init__(self, data, requires_grad=False):
        """
        Initialize a tensor with data. Data can be a number, list, or numpy array.
        
        Args:
            data: The data to initialize the tensor with
            requires_grad: Whether this tensor requires gradient computation
        """
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._children = []
        
        # Track which operation created this tensor (for visualization purposes)
        self._op_name = None
    
    @property
    def shape(self):
        """Return the shape of the tensor."""
        return self.data.shape
    
    def __repr__(self):
        """String representation of the tensor."""
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_str})"
    
    def __add__(self, other):
        """Add two tensors."""
        other_data = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data + other_data)
        
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._children = [(self, 1.0), (other, 1.0)] if isinstance(other, Tensor) else [(self, 1.0)]
            result._op_name = "add"
            
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    self.grad += result.grad
                
                if isinstance(other, Tensor) and other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data)
                    other.grad += result.grad
            
            result.grad_fn = _backward
        
        return result
    
    def __mul__(self, other):
        """Multiply two tensors element-wise."""
        other_data = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data * other_data)
        
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._children = [(self, other_data), (other, self.data)] if isinstance(other, Tensor) else [(self, other)]
            result._op_name = "mul"
            
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    
                    if isinstance(other, Tensor):
                        self.grad += other.data * result.grad
                    else:
                        self.grad += other * result.grad
                
                if isinstance(other, Tensor) and other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data)
                    other.grad += self.data * result.grad
            
            result.grad_fn = _backward
        
        return result
    
    def __pow__(self, power):
        """Raise tensor to a power."""
        result = Tensor(self.data ** power)
        
        if self.requires_grad:
            result.requires_grad = True
            result._children = [(self, power)]
            result._op_name = "pow"
            
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    self.grad += (power * self.data ** (power - 1)) * result.grad
            
            result.grad_fn = _backward
        
        return result
    
    def sum(self):
        """Sum all elements in the tensor."""
        result = Tensor(np.sum(self.data))
        
        if self.requires_grad:
            result.requires_grad = True
            result._children = [(self, 1.0)]
            result._op_name = "sum"
            
            def _backward():
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.ones_like(self.data) * result.grad
            
            result.grad_fn = _backward
        
        return result
    
    def matmul(self, other):
        """Matrix multiplication."""
        assert isinstance(other, Tensor), "Other must be a Tensor"
        result = Tensor(np.matmul(self.data, other.data))
        
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._children = [(self, 'matmul_left'), (other, 'matmul_right')]
            result._op_name = "matmul"
            
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    self.grad += np.matmul(result.grad, other.data.T)
                
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data)
                    other.grad += np.matmul(self.data.T, result.grad)
            
            result.grad_fn = _backward
        
        return result
    
    def backward(self):
        """Compute gradients via backpropagation."""
        # Initialize gradient at the output
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # Topological ordering of all nodes
        topo = []
        visited = set()
        
        def build_topo(node):
            if node in visited:
                return
            visited.add(node)
            
            if hasattr(node, '_children'):
                for child, _ in node._children:
                    if isinstance(child, Tensor):
                        build_topo(child)
            topo.append(node)
        
        build_topo(self)
        
        # Backward pass
        for node in reversed(topo):
            if node.grad_fn:
                node.grad_fn()
    
    def to_numpy(self):
        """Convert tensor to numpy array."""
        return self.data.copy()

    @classmethod
    def zeros(cls, shape, requires_grad=False):
        """Create a tensor of zeros with the given shape."""
        return cls(np.zeros(shape), requires_grad=requires_grad)
    
    @classmethod
    def ones(cls, shape, requires_grad=False):
        """Create a tensor of ones with the given shape."""
        return cls(np.ones(shape), requires_grad=requires_grad)
    
    @classmethod
    def randn(cls, *shape, requires_grad=False):
        """Create a tensor of random values with the given shape."""
        return cls(np.random.randn(*shape), requires_grad=requires_grad)


def demo_tensor_basics():
    """Demonstrate basic tensor operations."""
    print("===== Tensor Basics =====")
    # Create tensors
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    # Basic operations
    c = a + b
    d = a * b
    
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"a + b: {c}")
    print(f"a * b: {d}")
    print(f"Shape of a: {a.shape}")
    
    # Multi-dimensional tensors
    matrix_a = Tensor([[1, 2], [3, 4]])
    matrix_b = Tensor([[5, 6], [7, 8]])
    
    print(f"\nMatrix A:\n{matrix_a}")
    print(f"Shape of Matrix A: {matrix_a.shape}")
    print(f"\nMatrix B:\n{matrix_b}")
    print(f"Shape of Matrix B: {matrix_b.shape}")
    
    # Matrix multiplication
    result = matrix_a.matmul(matrix_b)
    print(f"\nMatrix A @ Matrix B:\n{result}")


if __name__ == "__main__":
    demo_tensor_basics()
