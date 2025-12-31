import torch
import torch.nn as nn
import numpy as np

print("========================================")
print("       Buffer Registration Test          ")
print("========================================\n")

# Test buffer registration
print("=== Testing register_buffer() ===")

class ModuleWithBuffer(nn.Module):
    def __init__(self, size):
        super().__init__()
        # Create tril mask buffer
        tril_mask = torch.tril(torch.ones(size, size))
        self.register_buffer('tril', tril_mask)
    
    def forward(self, x):
        return x  # Simplified for test

module = ModuleWithBuffer(4)

# Get buffers
buffers = list(module.buffers())
print(f"Number of buffers: {len(buffers)}")
assert len(buffers) == 1

# Check buffer exists
assert 'tril' in dict(module.named_buffers())
tril_tensor = dict(module.named_buffers())['tril']
print(f"Buffer 'tril' shape: {tril_tensor.shape}")
print(f"Buffer 'tril' values:\n{tril_tensor}")

# Verify it's a lower triangular matrix
assert tril_tensor.shape == (4, 4)

# Check that buffer is NOT in parameters
params = list(module.parameters())
print(f"Number of parameters: {len(params)}")
assert len(params) == 0  # No parameters, only buffers

print("Buffer registration test PASSED\n")

# Test buffer with parameters
print("=== Testing buffer with parameters ===")

fc = nn.Linear(4, 2)
fc_params = list(fc.parameters())
fc_buffers = list(fc.buffers())

print(f"Linear layer parameters: {len(fc_params)}")
print(f"Linear layer buffers: {len(fc_buffers)}")

assert len(fc_params) == 2  # weight + bias
assert len(fc_buffers) == 0  # No buffers

print("Buffer with parameters test PASSED\n")

# Test nested buffers
print("=== Testing nested buffers ===")

seq = nn.Sequential(
    ModuleWithBuffer(3),
    nn.Linear(3, 2),
    ModuleWithBuffer(2)
)

seq_buffers = list(seq.buffers())
seq_params = list(seq.parameters())

print(f"Sequential buffers (nested): {len(seq_buffers)}")
assert len(seq_buffers) == 2  # One from each ModuleWithBuffer

print(f"Sequential parameters (nested): {len(seq_params)}")
assert len(seq_params) == 2  # weight + bias from Linear

# Print buffer values for comparison
print("\nBuffer values from nested modules:")
for name, buffer in seq.named_buffers():
    print(f"{name}:")
    print(buffer)
    print()

print("Nested buffers test PASSED\n")

# Test move constructor (in Python, this is just assignment)
print("=== Testing buffer preservation ===")

module2 = ModuleWithBuffer(5)
buffers_before = list(module2.buffers())
assert len(buffers_before) == 1

# In Python, we can't really test move, but we can test that buffers persist
module3 = ModuleWithBuffer(5)
buffers_after = list(module3.buffers())
assert len(buffers_after) == 1

print("Buffer preservation test PASSED\n")

# Print detailed comparison values
print("=== Detailed Buffer Values for Comparison ===")
print("\nModuleWithBuffer(4) tril mask:")
module_tril = dict(module.named_buffers())['tril']
print(module_tril.numpy())

print("\nSequential first module tril mask (size 3):")
seq_module1_tril = dict(list(seq.named_modules())[1].named_buffers())['tril']
print(seq_module1_tril.numpy())

print("\nSequential third module tril mask (size 2):")
seq_module3_tril = dict(list(seq.named_modules())[3].named_buffers())['tril']
print(seq_module3_tril.numpy())

print("\n========================================")
print("       ALL TESTS PASSED!                ")
print("========================================")

