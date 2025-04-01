import numpy as np
import matplotlib.pyplot as plt
import time

# This is a simulation of GPU acceleration since we're implementing our own system
# In a real scenario, you would use PyTorch's GPU capabilities

class Device:
    """Base class for compute devices."""
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return self.name


# Define CPU and GPU devices
cpu = Device("cpu")
gpu = Device("gpu")


class Tensor:
    """
    Enhanced tensor with device support to simulate GPU acceleration.
    This is a simplified version for demonstration purposes.
    """
    
    def __init__(self, data, device=cpu, requires_grad=False):
        """
        Initialize a tensor with data on a specific device.
        
        Args:
            data: The data to initialize the tensor with
            device: The compute device (cpu or gpu)
            requires_grad: Whether this tensor requires gradient computation
        """
        self.data = np.array(data, dtype=np.float32)
        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        
        # For demonstration: simulate memory transfer if device is GPU
        if device == gpu:
            # In a real implementation, this would be the actual time to transfer data to GPU
            # Here we just simulate a small delay
            time.sleep(0.0001 * (self.data.size / 1000))
    
    def to(self, device):
        """Move tensor to another device."""
        if self.device == device:
            return self
        
        # Create a new tensor on the target device
        result = Tensor(self.data, device=device, requires_grad=self.requires_grad)
        
        # Copy gradient if it exists
        if self.grad is not None:
            result.grad = self.grad.copy()
        
        return result
    
    def cpu(self):
        """Move tensor to CPU."""
        return self.to(cpu)
    
    @property
    def shape(self):
        """Return the shape of the tensor."""
        return self.data.shape
    
    def __repr__(self):
        """String representation of the tensor."""
        device_str = f", device='{self.device}'"
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{device_str}{grad_str})"
    
    def __matmul__(self, other):
        """Matrix multiplication with GPU acceleration simulation."""
        assert isinstance(other, Tensor), "Other must be a Tensor"
        assert self.device == other.device, "Both tensors must be on the same device"
        
        start_time = time.time()
        
        # Simulate different computation times based on device
        if self.device == gpu:
            # GPU is much faster for large matrices
            delay_factor = 0.01  # Smaller delay for GPU
        else:
            # CPU is slower
            delay_factor = 0.1  # Larger delay for CPU
        
        # Size-based delay to simulate computation time
        matrix_size = self.data.size * other.data.size
        time.sleep(delay_factor * (matrix_size / 10000))
        
        # Actual computation
        result_data = np.matmul(self.data, other.data)
        
        # Create result tensor on the same device
        result = Tensor(result_data, device=self.device)
        
        end_time = time.time()
        
        # For demo purposes, print computation time
        #print(f"Matrix multiplication on {self.device} took {end_time - start_time:.6f} seconds")
        
        return result


def benchmark_matrix_multiplication(sizes, num_trials=3):
    """
    Benchmark matrix multiplication across different sizes on CPU and GPU.
    
    Args:
        sizes: List of matrix sizes to test
        num_trials: Number of trials for each size
        
    Returns:
        Dictionary of benchmark results
    """
    results = {
        'sizes': sizes,
        'cpu_times': [],
        'gpu_times': [],
        'speedups': []
    }
    
    for size in sizes:
        print(f"Benchmarking size {size}x{size}...")
        
        # CPU benchmarking
        cpu_times = []
        for _ in range(num_trials):
            A_cpu = Tensor(np.random.randn(size, size), device=cpu)
            B_cpu = Tensor(np.random.randn(size, size), device=cpu)
            
            start_time = time.time()
            _ = A_cpu @ B_cpu
            end_time = time.time()
            
            cpu_times.append(end_time - start_time)
        
        avg_cpu_time = sum(cpu_times) / num_trials
        
        # GPU benchmarking
        gpu_times = []
        for _ in range(num_trials):
            A_gpu = Tensor(np.random.randn(size, size), device=gpu)
            B_gpu = Tensor(np.random.randn(size, size), device=gpu)
            
            start_time = time.time()
            _ = A_gpu @ B_gpu
            end_time = time.time()
            
            gpu_times.append(end_time - start_time)
        
        avg_gpu_time = sum(gpu_times) / num_trials
        
        # Calculate speedup
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else float('inf')
        
        results['cpu_times'].append(avg_cpu_time)
        results['gpu_times'].append(avg_gpu_time)
        results['speedups'].append(speedup)
        
        print(f"  CPU time: {avg_cpu_time:.6f}s, GPU time: {avg_gpu_time:.6f}s, Speedup: {speedup:.2f}x")
    
    return results


def visualize_benchmark(results):
    """Visualize benchmark results."""
    sizes = results['sizes']
    cpu_times = results['cpu_times']
    gpu_times = results['gpu_times']
    speedups = results['speedups']
    
    # Plot computation times
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(sizes, cpu_times, 'o-', label='CPU')
    plt.plot(sizes, gpu_times, 'o-', label='GPU')
    plt.title('Matrix Multiplication Computation Time')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot speedup
    plt.subplot(2, 1, 2)
    plt.plot(sizes, speedups, 'o-', color='green')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    plt.title('GPU Speedup Relative to CPU')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Factor (CPU Time / GPU Time)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gpu_benchmark.png')
    plt.close()


def demo_move_to_device():
    """Demonstrate moving tensors between devices."""
    print("\n===== Moving Tensors Between Devices =====")
    
    # Create a tensor on CPU
    x_cpu = Tensor([1, 2, 3, 4], device=cpu)
    print(f"Original tensor: {x_cpu}")
    
    # Move to GPU
    x_gpu = x_cpu.to(gpu)
    print(f"After moving to GPU: {x_gpu}")
    
    # Move back to CPU
    x_back_to_cpu = x_gpu.cpu()
    print(f"After moving back to CPU: {x_back_to_cpu}")
    
    # Try to perform operations across devices (should raise an error)
    try:
        result = x_cpu @ x_gpu
        print("Operation succeeded (should not happen)")
    except AssertionError as e:
        print(f"Expected error when operating across devices: {e}")


def moveTo(obj, device):
    """
    Move an object or collection of objects to the specified device.
    
    Args:
        obj: The object to move
        device: The target device
        
    Returns:
        The object on the target device
    """
    if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        return {moveTo(k, device): moveTo(v, device) for k, v in obj.items()}
    elif hasattr(obj, 'to'):
        return obj.to(device)
    else:
        return obj


def demo_move_collections():
    """Demonstrate moving collections of tensors between devices."""
    print("\n===== Moving Collections of Tensors =====")
    
    # Create a collection of tensors on CPU
    tensors = [
        Tensor([1, 2]), 
        Tensor([3, 4]),
        {"key": Tensor([5, 6]), "nested": [Tensor([7, 8]), Tensor([9, 10])]}
    ]
    
    print("Original collection:")
    print(tensors)
    
    # Move all tensors to GPU
    gpu_tensors = moveTo(tensors, gpu)
    
    print("\nAfter moving to GPU:")
    print(gpu_tensors)
    
    # Move back to CPU
    cpu_tensors = moveTo(gpu_tensors, cpu)
    
    print("\nAfter moving back to CPU:")
    print(cpu_tensors)


def demo_gpu_acceleration():
    """Demonstrate GPU acceleration for different matrix sizes."""
    print("\n===== GPU Acceleration Benchmark =====")
    
    # Benchmark small, medium, and large matrices
    sizes = [10, 50, 100, 200, 500, 1000]
    results = benchmark_matrix_multiplication(sizes)
    
    # Visualize results
    visualize_benchmark(results)
    print("\nBenchmark visualization saved as 'gpu_benchmark.png'")
    
    # Find the size where GPU becomes faster than CPU
    crossover_idx = next((i for i, speedup in enumerate(results['speedups']) if speedup > 1), None)
    
    if crossover_idx is not None:
        crossover_size = sizes[crossover_idx]
        print(f"\nGPU becomes faster than CPU at matrix size: {crossover_size}x{crossover_size}")
    else:
        print("\nGPU did not show speedup for the tested sizes")


if __name__ == "__main__":
    demo_move_to_device()
    demo_move_collections()
    demo_gpu_acceleration()
