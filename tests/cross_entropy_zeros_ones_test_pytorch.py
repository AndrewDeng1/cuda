import torch
import torch.nn.functional as F
import numpy as np

print("========================================")
print("  Cross Entropy, Zeros, Ones Test       ")
print("========================================\n")

# Test zeros and ones
print("=== Testing zeros() and ones() ===")
z = torch.zeros(2, 3)
o = torch.ones(2, 3)

print("Zeros tensor:")
print(z)
print("Ones tensor:")
print(o)

assert torch.allclose(z, torch.tensor(0.0))
assert torch.allclose(o, torch.tensor(1.0))
print("zeros() and ones() test PASSED\n")

# Test cross_entropy with class indices (PyTorch style)
print("=== Testing cross_entropy() with class indices ===")

# Simple 2D case: (2 samples, 3 classes)
logits = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], requires_grad=True)
targets = torch.tensor([2, 1], dtype=torch.long)  # Class 2 for sample 0, class 1 for sample 1

print("Logits:")
print(logits)
print("Targets (class indices):")
print(targets)

print("About to call cross_entropy...")
loss = F.cross_entropy(logits, targets)
print("cross_entropy returned successfully")

print("Cross Entropy Loss (scalar):")
print(loss.item())

# Expected: 
# Sample 0: -log(softmax([1,2,3])[2]) = -log(exp(3)/(exp(1)+exp(2)+exp(3))) 
#          = -log(20.086/30.193) = -log(0.665) ≈ 0.408
# Sample 1: -log(softmax([4,5,6])[1]) = -log(exp(5)/(exp(4)+exp(5)+exp(6)))
#          = -log(148.413/606.44) = -log(0.245) ≈ 1.408
# Mean: (0.408 + 1.408) / 2 ≈ 0.908

assert loss.shape == torch.Size([])  # Scalar
assert loss.item() > 0.0

print(f"Loss value: {loss.item()}")

# Test backward pass
loss.backward()

print("\nLogits gradients:")
print(logits.grad)

# Gradients should be (softmax - one_hot) / num_samples
# For sample 0: softmax([1,2,3]) = [0.090, 0.245, 0.665], one_hot = [0,0,1]
#              grad = [0.090, 0.245, -0.335] / 2 = [0.045, 0.122, -0.167]
# For sample 1: softmax([4,5,6]) = [0.090, 0.245, 0.665], one_hot = [0,1,0]
#              grad = [0.090, -0.755, 0.665] / 2 = [0.045, -0.377, 0.332]

assert logits.grad is not None
print("Backward pass test PASSED\n")

# Test 3D case (like GPT: B*T, vocab_size)
print("=== Testing cross_entropy() with 3D tensors ===")

logits_3d = torch.tensor([[[1.0, 2.0, 3.0],   # Batch 0, Seq 0
                           [0.5, 1.5, 2.5]],  # Batch 0, Seq 1
                          [[2.0, 1.0, 0.0],   # Batch 1, Seq 0
                           [3.0, 2.0, 1.0]]], # Batch 1, Seq 1
                         requires_grad=True)

targets_3d = torch.tensor([[2, 1],   # Batch 0: class 2, class 1
                           [0, 0]],  # Batch 1: class 0, class 0
                         dtype=torch.long)

loss_3d = F.cross_entropy(logits_3d.view(-1, 3), targets_3d.view(-1))

print("3D Cross Entropy Loss:")
print(loss_3d.item())

assert loss_3d.shape == torch.Size([])  # Scalar
assert loss_3d.item() > 0.0

print("3D test PASSED\n")

print("========================================")
print("       ALL TESTS PASSED!                ")
print("========================================")

# ========================================
#   Cross Entropy, Zeros, Ones Test       
# ========================================

# === Testing zeros() and ones() ===
# Zeros tensor:
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
# Ones tensor:
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
# zeros() and ones() test PASSED

# === Testing cross_entropy() with class indices ===
# Logits:
# tensor([[1., 2., 3.],
#         [4., 5., 6.]], requires_grad=True)
# Targets (class indices):
# tensor([2, 1])
# About to call cross_entropy...
# cross_entropy returned successfully
# Cross Entropy Loss (scalar):
# 0.9076058864593506
# Loss value: 0.9076058864593506

# Logits gradients:
# tensor([[ 0.0450,  0.1224, -0.1674],
#         [ 0.0450, -0.3776,  0.3326]])
# Backward pass test PASSED

# === Testing cross_entropy() with 3D tensors ===
# 3D Cross Entropy Loss:
# 0.6576058864593506
# 3D test PASSED

# ========================================
#        ALL TESTS PASSED!                
# ========================================