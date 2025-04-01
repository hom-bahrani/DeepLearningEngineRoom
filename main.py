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
from tensor_basics import demo_tensor_basics

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


if __name__ == "__main__":
    main()
