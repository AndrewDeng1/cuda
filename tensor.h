#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class Tensor : public enable_shared_from_this<Tensor> {
    
    public:
        Tensor();
        Tensor(const vector<int>& shape, bool requires_grad=false);
        Tensor(const vector<int>& shape, const vector<float>& data, bool requires_grad=false);
        Tensor(const vector<int>& shape, float* data, bool requires_grad=false);
        Tensor(shared_ptr<Tensor> other);

        // // Member functions
        int size();
        // void backward();
        float& at(vector<int> indices);
        float& at(int index);
        vector<int> compute_strides(const vector<int>& shape);
        shared_ptr<Tensor> reshape(const vector<int>& new_shape);
        shared_ptr<Tensor> reduce_to_shape(const vector<int>& target_shape);
        shared_ptr<Tensor> sum(int axis, bool keepdims=false);
        shared_ptr<Tensor> transpose(int dim1, int dim2);
        void backward();
        void print();

        // Broadcasting and reduction operations
        shared_ptr<Tensor> broadcast(const vector<int>& new_shape, bool matmul = false);

        // // Member variables
        bool requires_grad;
        vector<int> shape;
        vector<int> strides;
        float* data;
        shared_ptr<Tensor> grad;
        vector<shared_ptr<Tensor>> parents;
        function<void()> backward_fn;
};

// // Free function declarations
shared_ptr<Tensor> operator+(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator*(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator/(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor>& operator+=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor>& operator-=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor>& operator*=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor>& operator/=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);

// Scalar-tensor operators
shared_ptr<Tensor> operator+(const shared_ptr<Tensor>& A, float B);
shared_ptr<Tensor> operator+(float A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A, float B);
shared_ptr<Tensor> operator-(float A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A);
shared_ptr<Tensor> operator*(const shared_ptr<Tensor>& A, float B);
shared_ptr<Tensor> operator*(float A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator/(const shared_ptr<Tensor>& A, float B);
shared_ptr<Tensor> operator/(float A, const shared_ptr<Tensor>& B);

// Global functions
bool is_broadcastable(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul = false);
vector<int> get_broadcast_shape(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul = false);

#endif // TENSOR_H