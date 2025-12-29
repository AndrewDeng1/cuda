// // FOR WHEN I RUN STUFF ON MAC


// #include "tensor_kernels.h"
// #include <stdexcept>

// // Stub implementations for CPU-only builds (no CUDA)
// // These will throw if accidentally called

// void launch_add(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_subtract(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_multiply(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_divide(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_broadcast(shared_ptr<Tensor> a, shared_ptr<Tensor> b, vector<int>& padded_shape, vector<int>& padded_strides, bool matmul) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_sum(shared_ptr<Tensor> a, shared_ptr<Tensor> b, int axis) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_transpose(shared_ptr<Tensor> a, shared_ptr<Tensor> b, int dim1, int dim2) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_pow(shared_ptr<Tensor> a, shared_ptr<Tensor> b, int exponent) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_relu(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_sigmoid(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_tanh(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_softmax(shared_ptr<Tensor> a, shared_ptr<Tensor> sm_exp, shared_ptr<Tensor> sm_exp_broadcast, shared_ptr<Tensor> b, int axis) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_matmul(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> c) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

// void launch_cross_entropy(shared_ptr<Tensor> logits, shared_ptr<Tensor> y_true, shared_ptr<Tensor> result, int axis) {
//     throw std::runtime_error("CUDA not available - use CPU device");
// }

