"""
Deep Learning Foundations - Main Script

This script demonstrates the core concepts of deep learning.

Run this script to see demonstrations of:
1. Tensor operations
2. Automatic differentiation
3. Gradient descent optimization
4. Dataset loading
5. Neural network training
6. GPU acceleration (simulated)
"""

import os
import time

# Import our modules
from src.tensor_basics import demo_tensor_basics
from src.autograd import (
    demo_automatic_differentiation, 
    demo_gradient_descent, 
    demo_optimization_with_autograd,
    demo_multi_variable_optimization
)
from src.datasets import demo_dataset_loading
from src.neural_network import demo_neural_network
from src.gpu import demo_gpu_acceleration

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50 + "\n")

def main():
    """Run all demonstrations."""
    print_header("DEEP LEARNING FOUNDATIONS")
    print("This program demonstrates the core concepts of deep learning")
    print("by implementing them from scratch in Python.")
    print("\nPress Enter to continue through each demonstration...")
    
    # Tensor Basics
    input("\nPress Enter to start Tensor Basics demonstration...")
    print_header("1. TENSOR BASICS")
    demo_tensor_basics()

    # Automatic Differentiation
    input("\nPress Enter to start Automatic Differentiation demonstration...")
    print_header("2. AUTOMATIC DIFFERENTIATION")
    demo_automatic_differentiation()
    
    # Gradient Descent
    input("\nPress Enter to start Gradient Descent demonstration...")
    print_header("3. GRADIENT DESCENT")
    demo_gradient_descent()
    
    # Optimization with Automatic Differentiation
    input("\nPress Enter to start Optimization with Automatic Differentiation demonstration...")
    print_header("4. OPTIMIZATION WITH AUTOMATIC DIFFERENTIATION")
    demo_optimization_with_autograd()
    
    # Multi-Variable Optimization
    input("\nPress Enter to start Multi-Variable Optimization demonstration...")
    print_header("5. MULTI-VARIABLE OPTIMIZATION")
    demo_multi_variable_optimization()
    
    # Dataset Loading
    input("\nPress Enter to start Dataset Loading demonstration...")
    print_header("6. DATASET LOADING")
    demo_dataset_loading()

    # Neural Network Training
    input("\nPress Enter to start Neural Network Training demonstration...")
    print_header("7. NEURAL NETWORK TRAINING")
    demo_neural_network()

    # GPU Acceleration
    input("\nPress Enter to start GPU Acceleration demonstration...")
    print_header("8. GPU ACCELERATION")
    demo_gpu_acceleration()
    
    print("\nAll demonstrations complete. Thank you for exploring Deep Learning Foundations!")


if __name__ == "__main__":
    main()
