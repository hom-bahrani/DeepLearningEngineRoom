#!/usr/bin/env python
"""
GPU Acceleration Test Tool for Mac M1

This script tests the GPU acceleration capabilities on Mac M1 computers
using TensorFlow and Metal.
"""

import os
import sys
import subprocess
import platform
import time

def check_python_env():
    """Print information about the current Python environment."""
    print("\n===== Python Environment =====")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {platform.system()}")
    print(f"Processor: {platform.processor()}")

def check_tensorflow():
    """Check TensorFlow installation and GPU availability."""
    print("\n===== TensorFlow Status =====")
    
    # Try importing TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        
        # Check for GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU device(s):")
            for gpu in gpus:
                print(f"  {gpu}")
            return True
        else:
            print("No GPU devices detected by TensorFlow")
            return False
    
    except ImportError:
        print("TensorFlow not installed in current Python environment")
        
        # Check if it might be installed in pyenv
        pyenv_path = os.path.expanduser("~/.pyenv/versions/3.9.6/bin/python")
        if os.path.exists(pyenv_path):
            print(f"Found pyenv Python at {pyenv_path}")
            print("Checking TensorFlow in pyenv...")
            
            try:
                cmd = [pyenv_path, "-c", 
                      "import tensorflow as tf; print(f'TensorFlow {tf.__version__} available with {len(tf.config.list_physical_devices(\"GPU\"))} GPU(s)')"]
                tf_output = subprocess.check_output(cmd).decode('utf-8').strip()
                print(f"From pyenv Python: {tf_output}")
                if "GPU(s)" in tf_output and not "0 GPU(s)" in tf_output:
                    print("GPU detected in pyenv Python!")
                    print(f"To use TensorFlow with GPU, run your scripts with: {pyenv_path}")
                    return True
            except Exception as e:
                print(f"Error checking TensorFlow in pyenv: {e}")
        
        return False

def run_gpu_benchmark():
    """Run a basic matrix multiplication benchmark."""
    print("\n===== GPU Benchmark =====")
    
    # Try using our GPU module
    try:
        from src.gpu import demo_gpu_acceleration
        print("Running GPU benchmark from src.gpu module...")
        demo_gpu_acceleration()
    except Exception as e:
        print(f"Error running GPU benchmark: {e}")
        
        # Try running with pyenv if available
        pyenv_path = os.path.expanduser("~/.pyenv/versions/3.9.6/bin/python")
        if os.path.exists(pyenv_path):
            print(f"Trying to run benchmark with {pyenv_path}...")
            try:
                subprocess.run([pyenv_path, "-c", "from src.gpu import demo_gpu_acceleration; demo_gpu_acceleration()"])
            except Exception as e:
                print(f"Error running benchmark with pyenv Python: {e}")

def run_tensorflow_test():
    """Run a simple TensorFlow benchmark test."""
    print("\n===== TensorFlow Benchmark =====")
    
    # Check if we need to use pyenv Python
    pyenv_path = os.path.expanduser("~/.pyenv/versions/3.9.6/bin/python")
    use_pyenv = False
    
    try:
        import tensorflow as tf
    except ImportError:
        if os.path.exists(pyenv_path):
            use_pyenv = True
        else:
            print("TensorFlow not available. Skipping TensorFlow benchmark.")
            return
    
    # Create a simple benchmark script
    test_script = """
import tensorflow as tf
import time
import numpy as np

# Enable Metal
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_METAL_DEVICE_ENABLE'] = '1'

# Print TensorFlow info
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices: {gpus}")

# Test sizes
sizes = [10, 100, 500, 1000, 2000]
trials = 3

results = {
    'sizes': sizes,
    'cpu_times': [],
    'gpu_times': [],
    'speedups': []
}

# Run benchmarks
for size in sizes:
    print(f"\\nBenchmarking {size}x{size} matrices...")
    
    # Generate random matrices
    a = np.random.random((size, size)).astype(np.float32)
    b = np.random.random((size, size)).astype(np.float32)
    
    # CPU benchmark
    cpu_times = []
    for _ in range(trials):
        with tf.device('/CPU:0'):
            tf_a = tf.constant(a)
            tf_b = tf.constant(b)
            
            start = time.time()
            result = tf.matmul(tf_a, tf_b)
            # Force execution to complete
            result.numpy()
            end = time.time()
            
            cpu_times.append(end - start)
    
    avg_cpu_time = sum(cpu_times) / trials
    
    # GPU benchmark if available
    if gpus:
        gpu_times = []
        for _ in range(trials):
            with tf.device('/GPU:0'):
                tf_a = tf.constant(a)
                tf_b = tf.constant(b)
                
                start = time.time()
                result = tf.matmul(tf_a, tf_b)
                # Force execution to complete
                result.numpy()
                end = time.time()
                
                gpu_times.append(end - start)
        
        avg_gpu_time = sum(gpu_times) / trials
        speedup = avg_cpu_time / avg_gpu_time
    else:
        avg_gpu_time = float('inf')
        speedup = 0
    
    results['cpu_times'].append(avg_cpu_time)
    results['gpu_times'].append(avg_gpu_time)
    results['speedups'].append(speedup)
    
    print(f"  CPU: {avg_cpu_time:.6f}s, GPU: {avg_gpu_time:.6f}s, Speedup: {speedup:.2f}x")

# Summary
print("\\n===== Summary =====")
print(f"{'Size':<10}{'CPU Time':<15}{'GPU Time':<15}{'Speedup':<10}")
for i, size in enumerate(sizes):
    print(f"{size:<10}{results['cpu_times'][i]:<15.6f}{results['gpu_times'][i]:<15.6f}{results['speedups'][i]:<10.2f}x")
"""
    
    # Save the script
    with open("tf_benchmark.py", "w") as f:
        f.write(test_script)
    
    # Run the script
    try:
        if use_pyenv:
            print(f"Running TensorFlow benchmark with {pyenv_path}...")
            subprocess.run([pyenv_path, "tf_benchmark.py"])
        else:
            print("Running TensorFlow benchmark...")
            exec(test_script)
    except Exception as e:
        print(f"Error running TensorFlow benchmark: {e}")
    
    # Clean up
    if os.path.exists("tf_benchmark.py"):
        os.remove("tf_benchmark.py")

def main():
    """Main function to run all tests."""
    print("===== Mac M1 GPU Acceleration Test =====")
    
    check_python_env()
    has_tf_gpu = check_tensorflow()
    
    if has_tf_gpu:
        run_tensorflow_test()
    
    # Always run the GPU benchmark which may use simulation if needed
    run_gpu_benchmark()
    
    print("\n===== Test Complete =====")
    if has_tf_gpu:
        print("✅ Your Mac M1 GPU is properly configured for acceleration!")
    else:
        print("⚠️ TensorFlow with Metal GPU acceleration not detected.")
        print("To enable GPU acceleration, make sure to install:")
        print("  - TensorFlow 2.13.0 or newer")
        print("  - tensorflow-metal 1.0.0 or newer")
        print("  - Run your code with a Python interpreter that has these packages installed")

if __name__ == "__main__":
    main() 