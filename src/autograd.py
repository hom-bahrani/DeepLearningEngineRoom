import numpy as np
import matplotlib.pyplot as plt
from .tensor_basics import Tensor

def f(x):
    """Example function: f(x) = (x - 2)²"""
    return (x - 2) ** 2

def df(x):
    """Analytical derivative of f(x) = (x - 2)²"""
    return 2 * (x - 2)

def visualize_function_and_derivative(f, df, x_range=(-7, 9), num_points=100):
    """Visualize a function and its derivative."""
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    f_values = [f(Tensor(x)).data for x in x_values]
    df_values = [df(Tensor(x)).data for x in x_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, f_values, label='f(x)')
    plt.plot(x_values, df_values, label='df/dx')
    plt.plot(x_values, [0] * len(x_values), 'k--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Function and its Derivative')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.savefig('function_and_derivative.png')
    plt.close()

def demo_automatic_differentiation():
    """Demonstrate automatic differentiation."""
    print("\n===== Automatic Differentiation =====")
    
    # Create a tensor with requires_grad=True
    x = Tensor(3.5, requires_grad=True)
    print(f"Initial x: {x}")
    
    # Compute the function value
    y = (x - 2) ** 2
    print(f"f(x) = (x - 2)² = {y}")
    
    # Compute gradients through backpropagation
    y.backward()
    print(f"Gradient df/dx at x=3.5: {x.grad}")
    print(f"Expected gradient (analytical): {df(Tensor(3.5)).data}")
    
    # More complex example
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    
    # f(a, b) = a² + a*b + b²
    c = a ** 2 + a * b + b ** 2
    print(f"\nComputing f(a, b) = a² + a*b + b² where a={a.data} and b={b.data}")
    print(f"Result: {c}")
    
    c.backward()
    print(f"Gradient df/da: {a.grad}")  # Should be 2*a + b = 2*2 + 3 = 7
    print(f"Gradient df/db: {b.grad}")  # Should be a + 2*b = 2 + 2*3 = 8


def gradient_descent(f, df_dx, x_start, learning_rate=0.1, num_steps=20):
    """
    Perform gradient descent to minimize a function.
    
    Args:
        f: The function to minimize
        df_dx: The derivative of the function
        x_start: Starting point
        learning_rate: Learning rate (step size)
        num_steps: Number of steps to take
        
    Returns:
        List of x values during optimization
        List of function values during optimization
    """
    x_history = [x_start]
    f_history = [f(x_start)]
    
    x = x_start
    for _ in range(num_steps):
        grad = df_dx(x)
        x = x - learning_rate * grad
        x_history.append(x)
        f_history.append(f(x))
    
    return x_history, f_history


def visualize_gradient_descent(f, x_history, f_history, x_range=(-1, 5), num_points=100):
    """Visualize the gradient descent process."""
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    f_values = [f(x) for x in x_values]
    
    plt.figure(figsize=(10, 6))
    
    # Plot the function
    plt.plot(x_values, f_values, 'b-', alpha=0.5, label='f(x)')
    
    # Plot the gradient descent steps
    plt.plot(x_history, f_history, 'ro-', alpha=0.7, label='Gradient Descent Path')
    
    # Highlight starting and ending points
    plt.scatter([x_history[0]], [f_history[0]], color='g', s=100, label='Starting Point')
    plt.scatter([x_history[-1]], [f_history[-1]], color='purple', s=100, label='Final Point')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Gradient Descent Optimization')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig('gradient_descent.png')
    plt.close()


def demo_gradient_descent():
    """Demonstrate gradient descent optimization."""
    print("\n===== Gradient Descent Optimization =====")
    
    # Define a simple function to minimize: f(x) = (x - 2)²
    def f_simple(x):
        return (x - 2) ** 2
    
    def df_simple(x):
        return 2 * (x - 2)
    
    # Starting point
    x_start = -3.0
    
    print(f"Function to minimize: f(x) = (x - 2)²")
    print(f"Starting point: x = {x_start}")
    print(f"Analytical minimum: x = 2.0")
    
    # Perform gradient descent
    x_history, f_history = gradient_descent(f_simple, df_simple, x_start, learning_rate=0.1, num_steps=30)
    
    print(f"Final x after gradient descent: {x_history[-1]:.6f}")
    print(f"Final function value: {f_history[-1]:.6f}")
    
    # Visualize
    visualize_gradient_descent(f_simple, x_history, f_history)


def optimize_with_autograd(f, x_start, learning_rate=0.1, num_steps=20):
    """
    Perform optimization using automatic differentiation.
    
    Args:
        f: The function to minimize (takes a Tensor and returns a Tensor)
        x_start: Starting point (float)
        learning_rate: Learning rate (step size)
        num_steps: Number of steps to take
        
    Returns:
        List of x values during optimization
        List of function values during optimization
    """
    x_history = [x_start]
    f_history = [f(Tensor(x_start)).data.item()]
    
    x = Tensor(x_start, requires_grad=True)
    
    for _ in range(num_steps):
        # Zero out previous gradients
        if x.grad is not None:
            x.grad = None
        
        # Forward pass
        y = f(x)
        
        # Backward pass
        y.backward()
        
        # Update x (we need to create a new tensor to avoid tracking history)
        with_grad_value = x.data - learning_rate * x.grad
        x = Tensor(with_grad_value, requires_grad=True)
        
        # Store history
        x_history.append(x.data.item())
        f_history.append(f(x).data.item())
    
    return x_history, f_history


def demo_optimization_with_autograd():
    """Demonstrate optimization using automatic differentiation."""
    print("\n===== Optimization with Automatic Differentiation =====")
    
    # Define a simple function to minimize: f(x) = (x - 2)²
    def f_tensor(x):
        return (x - Tensor(2.0)) ** 2
    
    # Starting point
    x_start = -3.0
    
    print(f"Function to minimize: f(x) = (x - 2)²")
    print(f"Starting point: x = {x_start}")
    print(f"Analytical minimum: x = 2.0")
    
    # Perform optimization with autograd
    x_history, f_history = optimize_with_autograd(f_tensor, x_start, learning_rate=0.1, num_steps=30)
    
    print(f"Final x after optimization: {x_history[-1]:.6f}")
    print(f"Final function value: {f_history[-1]:.6f}")
    
    # Visualize
    visualize_gradient_descent(lambda x: (x - 2)**2, x_history, f_history)


def demo_multi_variable_optimization():
    """Demonstrate optimization with multiple variables."""
    print("\n===== Multi-Variable Optimization =====")
    
    # Define a function of two variables: f(x, y) = (x - 2)² + (y - 3)²
    def f_multi(x, y):
        return (x - 2) ** 2 + (y - 3) ** 2
    
    # Create tensors with requires_grad=True
    x = Tensor(0.0, requires_grad=True)
    y = Tensor(0.0, requires_grad=True)
    
    print(f"Function to minimize: f(x, y) = (x - 2)² + (y - 3)²")
    print(f"Starting point: x = {x.data}, y = {y.data}")
    print(f"Analytical minimum: x = 2.0, y = 3.0")
    
    # Learning rate
    lr = 0.1
    
    # Store history
    history = []
    
    # Gradient descent
    for i in range(30):
        # Zero out previous gradients
        if x.grad is not None:
            x.grad = None
        if y.grad is not None:
            y.grad = None
        
        # Forward pass - compute function value
        z = (x - 2) ** 2 + (y - 3) ** 2
        
        # Store current values
        history.append((i, x.data.item(), y.data.item(), z.data.item()))
        
        # Backward pass - compute gradients
        z.backward()
        
        # Update parameters
        x_new = x.data - lr * x.grad
        y_new = y.data - lr * y.grad
        
        x = Tensor(x_new, requires_grad=True)
        y = Tensor(y_new, requires_grad=True)
    
    print(f"Final values: x = {x.data:.6f}, y = {y.data:.6f}")
    print(f"Final function value: {f_multi(x.data, y.data):.6f}")
    
    # Print optimization trajectory
    print("\nOptimization trajectory (selected steps):")
    for i, x_val, y_val, f_val in history[::5]:
        print(f"Step {i}: x = {x_val:.4f}, y = {y_val:.4f}, f(x,y) = {f_val:.4f}")


if __name__ == "__main__":
    demo_automatic_differentiation()
    demo_gradient_descent()
    demo_optimization_with_autograd()
    demo_multi_variable_optimization()
