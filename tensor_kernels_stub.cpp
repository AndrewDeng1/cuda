// FOR WHEN I RUN STUFF ON MAC


#include "tensor_kernels.h"
#include <stdexcept>

// Stub implementations for CPU-only builds (no CUDA)
// These will throw if accidentally called

void launch_add(const Tensor& a, const Tensor& b, Tensor& result) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_subtract(const Tensor& a, const Tensor& b, Tensor& result) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_multiply(const Tensor& a, const Tensor& b, Tensor& result) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_divide(const Tensor& a, const Tensor& b, Tensor& result) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_broadcast(const Tensor& a, Tensor& b, vector<int>& padded_shape, vector<int>& padded_strides, bool matmul) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_sum(const Tensor& a, Tensor& b, int axis) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_transpose(const Tensor& a, Tensor& b, int dim1, int dim2) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_pow(const Tensor& a, Tensor& b, int exponent) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_relu(const Tensor& a, Tensor& b) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_sigmoid(const Tensor& a, Tensor& b) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_tanh(const Tensor& a, Tensor& b) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_softmax(const Tensor& a, Tensor& sm_exp, Tensor& sm_exp_broadcast, Tensor& b, int axis) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    throw std::runtime_error("CUDA not available - use CPU device");
}

void launch_cross_entropy(const Tensor& logits, const Tensor& y_true, Tensor& result, int axis) {
    throw std::runtime_error("CUDA not available - use CPU device");
}
