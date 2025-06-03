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

        // Member functions
        int size();
        int numel() { return size(); }  // Alias for size()
        void backward();
        float& at(vector<int> indices);
        vector<int> compute_strides(const vector<int>& shape);
        shared_ptr<Tensor> reshape(const vector<int>& new_shape);
        shared_ptr<Tensor> sum(int axis, bool keepdims=false);

        // Member variables
        bool requires_grad;
        vector<int> shape;
        vector<int> strides;
        float* data;
        shared_ptr<Tensor> grad;
        vector<shared_ptr<Tensor>> parents;
        function<void()> backward_fn;
};

// Free function declarations
shared_ptr<Tensor> operator+(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator+=(shared_ptr<Tensor> A, const shared_ptr<Tensor>& B);

// Helper function declaration
shared_ptr<Tensor> reduce_to_shape(shared_ptr<Tensor> A, vector<int>& target_shape);

#endif // TENSOR_H