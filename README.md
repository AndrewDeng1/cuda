# BobbyLLM

A PyTorch-like neural network library implemented in C++ with CUDA support, designed to build modern transformers from scratch. This project provides a complete autograd system, neural network modules, optimizers, and tensor operations optimized for GPU acceleration.

## Features

### Tensor Operations (50+ operations)

The library provides a comprehensive set of tensor operations necessary to build modern transformers:

**Core Operations:**
- Arithmetic: `+`, `-`, `*`, `/` (element-wise and scalar operations)
- Matrix multiplication: `matmul()`
- Broadcasting: automatic shape broadcasting for compatible operations
- Reshape: `reshape()` with support for dimension inference (`-1`)
- Transpose: `transpose()` with negative index support
- Slicing: `slice()` for tensor indexing

**Reduction Operations:**
- `sum()`, `mean()`, `variance_squared()` with axis support and `keepdims` option

**Activation Functions:**
- `relu()`, `sigmoid()`, `tanh_op()`, `softmax()` (with axis support)

**Neural Network Operations:**
- `dropout()` with training/eval mode support
- `layer_norm()` with learnable gamma and beta parameters
- `embedding()` for token/position embeddings
- `cross_entropy()` loss function (PyTorch-style with class indices)

**Tensor Manipulation:**
- `cat()` - concatenate tensors along an axis
- `stack()` - stack tensors along a new dimension
- `masked_fill()` - fill masked elements with a value
- `tril()` - create lower triangular matrices (for causal attention masks)

**Random Operations:**
- `randn()` - sample from standard normal distribution
- `randint()` - sample random integers
- `multinomial()` - sample from multinomial distribution
- `xavier_normal()` - Xavier/Glorot initialization
- `kaiming_normal()` - Kaiming/He initialization

**Utility Functions:**
- `arange()` - create sequences of values
- `zeros()`, `ones()` - create constant tensors

**Automatic Differentiation:**
- Full backward pass support with automatic gradient computation
- Topological sort-based gradient accumulation
- Support for complex computation graphs

All operations support both CPU and CUDA execution (CUDA kernels available for key operations).

### Optimizers

Three optimizers are implemented with PyTorch-compatible APIs:

1. **SGD** - Stochastic Gradient Descent with optional momentum and weight decay
2. **Adam** - Adaptive Moment Estimation with learning rate, beta1, beta2, epsilon, and weight decay
3. **AdamW** - Adam with decoupled weight decay

All optimizers support:
- Parameter groups with different hyperparameters
- State management (momentum buffers, exponential moving averages)
- Gradient zeroing via `zero_grad()`
- Step-based optimization via `step()`

### Neural Network Modules

PyTorch-compatible module system with lazy registration:

1. **Linear** - Fully connected layer (`y = xW^T + b`) with optional bias
2. **ReLU** - Rectified Linear Unit activation
3. **Dropout** - Dropout regularization with training/eval mode support
4. **LayerNorm** - Layer normalization with learnable affine parameters
5. **Embedding** - Embedding lookup layer for token/position embeddings
6. **ModuleList** - Container for dynamically managing a list of modules
7. **Sequential** - Sequential container for applying modules in order

**Module Features:**
- Automatic parameter and buffer registration
- Recursive parameter collection from submodules
- Training/evaluation mode management
- Move semantics support with lazy registration
- Buffer support for non-trainable tensors (e.g., attention masks)

### Character-Level Encoding

The GPT implementation uses character-level tokenization:
- Automatic vocabulary creation from input text
- Character-to-index and index-to-character mappings
- Simple encoding/decoding utilities

### Example: GPT Language Model

The library includes a complete GPT (Generative Pre-trained Transformer) implementation (`gpt.cpp`) demonstrating:
- Multi-head self-attention with causal masking
- Transformer blocks with residual connections
- Position and token embeddings
- Language model head for next-token prediction
- Training loop with loss estimation
- Text generation with multinomial sampling

## Project Structure

```
.
├── tensor.h / tensor.cpp      # Core tensor operations and autograd
├── nn.h / nn.cpp              # Neural network modules
├── optim.h / optim.cpp        # Optimizers (SGD, Adam, AdamW)
├── tensor_kernels.cu          # CUDA kernel implementations
├── gpt.cpp                    # Full GPT model implementation
├── tests/                     # Comprehensive test suite
└── README.md                  # This file
```

## Building

The project uses a Makefile for compilation. CUDA support requires `nvcc` compiler.

```bash
make gpt    # Build the GPT model
make test   # Build and run tests
```

## Areas for Improvement

### Tokenization
- [ ] Support more tokenization methods (BPE, SentencePiece, WordPiece)
- [ ] Tokenizer abstraction layer

### CUDA Optimization
- [ ] Better CUDA kernels with tiling strategies
- [ ] Performance profiling and benchmarking
- [ ] More operations ported to CUDA
- [ ] Better parallelism (e.g., combine multiple tensors in attention operations)
- [ ] Flash Attention implementation for efficient attention computation

### Optimizers
- [ ] Support more optimizers (RMSprop, AdaGrad, AdaDelta, etc.)
- [ ] Learning rate schedulers

### Positional Encodings
- [ ] RoPE (Rotary Position Embedding)
- [ ] Sinusoidal positional encodings
- [ ] ALiBi (Attention with Linear Biases)

### Architecture Support
- [ ] CNN layers (Conv1D, Conv2D, etc.)
- [ ] RMSNorm (Root Mean Square Layer Normalization)
- [ ] Mixture of Experts (MoE)
- [ ] Additional activation functions (GELU, Swish, etc.)
- [ ] More normalization layers (BatchNorm, GroupNorm, etc.)

### Tensor Infrastructure
- [ ] Refactor tensor to use separate Storage object
- [ ] Enable slicing and view operations to reference same storage with different views
- [ ] Add offset/buffer to view for stride calculations
- [ ] Memory-efficient operations through shared storage

### Reinforcement Learning
- [ ] RLHF (Reinforcement Learning from Human Feedback) support
- [ ] Policy gradient methods
- [ ] Value function estimation

### Advanced Inference Features
- [ ] KV cache for efficient autoregressive generation
- [ ] Speculative decoding
- [ ] Temperature and top-k/top-p sampling
- [ ] Sliding window KV cache
- [ ] Attention sinks for long context
- [ ] Different attention architectures:
  - Multi-Query Attention (MQA)
  - Multi-Group Query Attention (MGQA)
  - Multi-Head Latent Attention (MHLA)

## Design Philosophy

This library follows PyTorch's design principles:
- **Eager execution** with dynamic computation graphs
- **Automatic differentiation** with reverse-mode autograd
- **Module-based architecture** for building neural networks
- **Lazy registration** for efficient move semantics
- **PyTorch-compatible APIs** for easy migration

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

