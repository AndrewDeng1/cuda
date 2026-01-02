#pragma once
#include "tensor.h"

void launch_add(const Tensor& a, const Tensor& b, Tensor& result);
void launch_subtract(const Tensor& a, const Tensor& b, Tensor& result);
void launch_multiply(const Tensor& a, const Tensor& b, Tensor& result);
void launch_divide(const Tensor& a, const Tensor& b, Tensor& result);
void launch_broadcast(const Tensor& a, Tensor& b, vector<int>& padded_shape, vector<int>& padded_strides, bool matmul);
void launch_sum(const Tensor& a, Tensor& b, int axis);
void launch_transpose(const Tensor& a, Tensor& b, int dim1, int dim2);
void launch_pow(const Tensor& a, Tensor& b, int exponent);
void launch_relu(const Tensor& a, Tensor& b);
void launch_sigmoid(const Tensor& a, Tensor& b);
void launch_tanh(const Tensor& a, Tensor& b);
void launch_softmax(const Tensor& a, Tensor& sm_exp, Tensor& sm_exp_broadcast, Tensor& b, int axis);
void launch_matmul(const Tensor& a, const Tensor& b, Tensor& c);
void launch_cross_entropy(const Tensor& logits, const Tensor& y_true, Tensor& result, int axis);