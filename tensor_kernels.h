#pragma once
#include "tensor.h"

void launch_add(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launch_subtract(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launch_multiply(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launch_divide(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launch_broadcast(shared_ptr<Tensor> a, shared_ptr<Tensor> b, vector<int>& padded_shape, vector<int>& padded_strides, bool matmul);
void launch_sum(shared_ptr<Tensor> a, shared_ptr<Tensor> b, int axis);
void launch_transpose(shared_ptr<Tensor> a, shared_ptr<Tensor> b, int dim1, int dim2);
void launch_pow(shared_ptr<Tensor> a, shared_ptr<Tensor> b, int exponent);
void launch_relu(shared_ptr<Tensor> a, shared_ptr<Tensor> b);
void launch_sigmoid(shared_ptr<Tensor> a, shared_ptr<Tensor> b);
void launch_tanh(shared_ptr<Tensor> a, shared_ptr<Tensor> b);
void launch_softmax(shared_ptr<Tensor> a, shared_ptr<Tensor> sm_exp, shared_ptr<Tensor> sm_exp_broadcast, shared_ptr<Tensor> b, int axis);
void launch_matmul(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> c);
void launch_cross_entropy(shared_ptr<Tensor> logits, shared_ptr<Tensor> y_true, shared_ptr<Tensor> result, int axis);