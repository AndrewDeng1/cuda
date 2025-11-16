#pragma once
#include "tensor.h"

void launch_add(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launch_subtract(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launch_multiply(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launch_divide(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launch_broadcast(shared_ptr<Tensor> a, shared_ptr<Tensor> b, vector<int>& padded_shape, vector<int>& padded_strides, bool matmul);
void launch_sum(shared_ptr<Tensor> a, shared_ptr<Tensor> b, int axis);