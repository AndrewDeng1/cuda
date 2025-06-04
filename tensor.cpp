#include "tensor.h"

Tensor::Tensor() : data(nullptr), grad(nullptr), parents(), backward_fn() {}

Tensor::Tensor(const vector<int>& shape, bool requires_grad) : shape(shape), requires_grad(requires_grad) {
    this->requires_grad = requires_grad;
    this->shape = shape;
    
    this->strides = compute_strides(shape);

    this->data = new float[size()];  // Changed from int to float

    if(requires_grad) {
        this->grad = make_shared<Tensor>(shape, false);
    } else {
        this->grad = nullptr;
    }

    this->parents = vector<shared_ptr<Tensor>>();
    this->backward_fn = [](){};
}

Tensor::Tensor(const vector<int>& shape, const vector<float>& data, bool requires_grad) : Tensor(shape, requires_grad) {
    
    if(data.size()!=size()){
        throw std::runtime_error("Data size mismatch");
    }

    for(int i=0; i<size(); i++){
        this->data[i] = data[i];
    }
}

Tensor::Tensor(shared_ptr<Tensor> other) : Tensor(other->shape, other->requires_grad) {
    this->data = other->data;
    this->grad = other->grad;
    this->parents = other->parents;
    this->backward_fn = other->backward_fn;
}

// Should enable [0, 0, 0, ..., d_n, d_n-1, ..., d_1, d_0]
// to access
// [d_n, d_n-1, ..., d_1, d_0]
float& Tensor::at(vector<int> indices){
    if(indices.size()<shape.size()){
        throw std::runtime_error("Indices size mismatch");
    }

    if(indices.size()>shape.size()){
        for(int i=0; i<indices.size()-shape.size(); i++){
            if(indices[i]!=0){
                throw std::runtime_error("Shape mismatch");
            }
        }
    }
    
    int ind=0;
    for(int i=indices.size()-shape.size(); i<indices.size(); i++){
        if(indices[i]>=shape[i-indices.size()+shape.size()]){
            throw std::runtime_error("Index out of bounds");
        }
        ind+=strides[i-indices.size()+shape.size()]*indices[i];
    }
    
    return data[ind];
}

int Tensor::size(){
    int cnt=1;
    for(int i=0; i<this->shape.size(); i++){
        cnt *= this->shape[i];
    }
    return cnt;
}

vector<int> Tensor::compute_strides(const vector<int>& shape) {
    vector<int> strides(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int>& new_shape) {
    int new_numel = 1;
    for (int dim : new_shape) new_numel *= dim;
    assert(new_numel == size());  // shapes must be compatible

    auto reshaped = std::make_shared<Tensor>(shared_from_this());  // shallow copy
    reshaped->shape = new_shape;
    reshaped->strides = compute_strides(new_shape);  // recompute strides
    return reshaped;
}

// void Tensor::backward() {
//     // Topological sort using Kahn's algorithm
//     unordered_map<shared_ptr<Tensor>, int> in_degree;
//     queue<shared_ptr<Tensor>> q;
//     unordered_set<shared_ptr<Tensor>> visited;
    
//     // First, traverse the entire graph to count in-degrees for all tensors
//     function<void(shared_ptr<Tensor>)> count_in_degrees = [&](shared_ptr<Tensor> tensor) {
//         if (visited.count(tensor)) return;
//         visited.insert(tensor);
        
//         // Count incoming edges for this tensor's parents
//         for (const auto& parent : tensor->parents) {
//             in_degree[parent]++;
//             count_in_degrees(parent);  // Recursively process parents
//         }
//     };
    
//     // Start counting from this tensor
//     count_in_degrees(shared_from_this());
    
//     // Add tensors with no incoming edges to queue
//     for (const auto& pair : in_degree) {
//         if (pair.second == 0) {
//             q.push(pair.first);
//         }
//     }
    
//     // Process tensors in topological order
//     while (!q.empty()) {
//         auto tensor = q.front();
//         q.pop();
        
//         // Call backward function if it exists
//         if (tensor->backward_fn) {
//             tensor->backward_fn();
//         }
        
//         // Update in-degree for children
//         for (const auto& child : tensor->parents) {
//             in_degree[child]--;
//             if (in_degree[child] == 0) {
//                 q.push(child);
//             }
//         }
//     }
// }

shared_ptr<Tensor> Tensor::sum(int axis, bool keepdims) {
    vector<int> new_shape;
    for(int i=0; i<shape.size(); i++){
        if(i==axis){
            if(keepdims){
                new_shape.push_back(1);
            }
        } else {
            new_shape.push_back(shape[i]);
        }
    }

    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, requires_grad);
    
    // Anonymous lambda function
    // Must capture itself, since needs to call itself recursively
    function<void(shared_ptr<Tensor>, shared_ptr<Tensor>, vector<int>, int, int)> rec = 
        [&](shared_ptr<Tensor> A, shared_ptr<Tensor> result, vector<int> acc, int axis, int ind) {
            if(ind==result->shape.size()){
                float val=0;
                vector<int> acc_A(acc);
                acc_A[axis]=0;
                for(int i=0; i<A->shape[axis]; i++){
                    val+=A->at(acc_A);
                    acc_A[axis]++;
                }
                result->at(acc)=val;
                return;
            }

            for(int i=0; i<result->shape[ind]; i++){
                vector<int> new_acc(acc);
                new_acc.push_back(i);
                rec(A, result, new_acc, axis, ind+1);
            }
        };

    rec(shared_from_this(), result, vector<int>(), axis, 0);
    return result;

    // NO BACKWARD FUNCTION IMPLEMENTED RN
}

shared_ptr<Tensor> Tensor::reduce_to_shape(const vector<int>& target_shape) {

    if(target_shape.size() > shape.size()){
        throw std::runtime_error("Target shape size mismatch");
    }
    
    vector<int> new_shape;
    if(target_shape.size() < shape.size()) {
        for(int i=0; i<shape.size()-target_shape.size(); i++) {
            new_shape.push_back(1);
        }
    }

    for(int i=0; i<target_shape.size(); i++) {
        new_shape.push_back(target_shape[i]);
    }

    shared_ptr<Tensor> result = make_shared<Tensor>(shared_from_this());
    for(int i=0; i<new_shape.size(); i++) {
        if(new_shape[i]==1 && shape[i]>1) {
            result = result->sum(i, true);
        } else if(new_shape[i] == shape[i]) {
            continue;
        } else {
            throw std::runtime_error("Target shape size mismatch");
        }
    }

    result = result->reshape(target_shape);
    return result;
}

shared_ptr<Tensor> operator+(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {

    vector<int> new_shape;
    int d = max(A->shape.size(), B->shape.size());
    for(int i=0; i<d; i++){
        if(i<d-A->shape.size()){
            new_shape.push_back(B->shape[i]);
        } else if(i<d-B->shape.size()){
            new_shape.push_back(A->shape[i]);
        } else {
            if(A->shape[i-(d-A->shape.size())]==1){
                new_shape.push_back(B->shape[i-(d-B->shape.size())]);
            } else if(B->shape[i-(d-B->shape.size())]==1){
                new_shape.push_back(A->shape[i-(d-A->shape.size())]);
            } else if(A->shape[i-(d-A->shape.size())]==B->shape[i-(d-B->shape.size())]){
                new_shape.push_back(A->shape[i-(d-A->shape.size())]);
            } else {
                throw invalid_argument("Shape mismatch");
            }
        }
    }

    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad||B->requires_grad);
    
    // Convert add to a lambda function
    function<void(shared_ptr<Tensor>, shared_ptr<Tensor>, shared_ptr<Tensor>, vector<int>&, vector<int>&, vector<int>, int)> add = 
        [&](shared_ptr<Tensor> A, shared_ptr<Tensor> B, shared_ptr<Tensor> C, vector<int>& dims_A, vector<int>& dims_B, vector<int> acc, int ind) {
            if(ind==dims_A.size()){
                vector<int> acc_A(acc);
                for(int i=0; i<acc_A.size(); i++){
                    if(dims_A[i]==1){
                        acc_A[i]=0;
                    }
                }
                vector<int> acc_B(acc);
                for(int i=0; i<acc_B.size(); i++){
                    if(dims_B[i]==1){
                        acc_B[i]=0;
                    }
                }
                C->at(acc)=A->at(acc_A)+B->at(acc_B);
                return;
            }

            for(int i=0; i<max(dims_A[ind], dims_B[ind]); i++){
                vector<int> new_acc(acc);
                new_acc.push_back(i);
                add(A, B, C, dims_A, dims_B, new_acc, ind+1);
            }
        };

    vector<int> dims_A(d);
    vector<int> dims_B(d);
    
    // Fill dims_A and dims_B with appropriate dimensions
    for(int i=0; i<d; i++) {
        if(i < d-A->shape.size()) {
            dims_A[i] = 1;
        } else {
            dims_A[i] = A->shape[i-(d-A->shape.size())];
        }
        
        if(i < d-B->shape.size()) {
            dims_B[i] = 1;
        } else {
            dims_B[i] = B->shape[i-(d-B->shape.size())];
        }
    }
    
    add(A, B, result, dims_A, dims_B, vector<int>(), 0);

    if(A->requires_grad || B->requires_grad){
        result->backward_fn = [A, B, result](){
            if(A->requires_grad && A->grad!=nullptr){
                auto reduced_grad = result->grad->reduce_to_shape(A->shape);
                reduced_grad->requires_grad = false;
                A->grad = A->grad + reduced_grad;
            }
            if(B->requires_grad && B->grad!=nullptr){
                auto reduced_grad = result->grad->reduce_to_shape(B->shape);
                reduced_grad->requires_grad = false;
                B->grad = B->grad + reduced_grad;
            }
        };
    }
    return result;
}

shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    vector<int> new_shape;
    int d = max(A->shape.size(), B->shape.size());
    for(int i=0; i<d; i++){
        if(i<d-A->shape.size()){
            new_shape.push_back(B->shape[i]);
        } else if(i<d-B->shape.size()){
            new_shape.push_back(A->shape[i]);
        } else {
            if(A->shape[i-(d-A->shape.size())]==1){
                new_shape.push_back(B->shape[i-(d-B->shape.size())]);
            } else if(B->shape[i-(d-B->shape.size())]==1){
                new_shape.push_back(A->shape[i-(d-A->shape.size())]);
            } else if(A->shape[i-(d-A->shape.size())]==B->shape[i-(d-B->shape.size())]){
                new_shape.push_back(A->shape[i-(d-A->shape.size())]);
            } else {
                throw invalid_argument("Shape mismatch");
            }
        }
    }

    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad||B->requires_grad);
    
    // Convert subtract to a lambda function
    function<void(shared_ptr<Tensor>, shared_ptr<Tensor>, shared_ptr<Tensor>, vector<int>&, vector<int>&, vector<int>, int)> subtract = 
        [&](shared_ptr<Tensor> A, shared_ptr<Tensor> B, shared_ptr<Tensor> C, vector<int>& dims_A, vector<int>& dims_B, vector<int> acc, int ind) {
            if(ind==dims_A.size()){
                vector<int> acc_A(acc);
                for(int i=0; i<acc_A.size(); i++){
                    if(dims_A[i]==1){
                        acc_A[i]=0;
                    }
                }
                vector<int> acc_B(acc);
                for(int i=0; i<acc_B.size(); i++){
                    if(dims_B[i]==1){
                        acc_B[i]=0;
                    }
                }
                C->at(acc)=A->at(acc_A)-B->at(acc_B);
                return;
            }

            for(int i=0; i<max(dims_A[ind], dims_B[ind]); i++){
                vector<int> new_acc(acc);
                new_acc.push_back(i);
                add(A, B, C, dims_A, dims_B, new_acc, ind+1);
            }
        };

    vector<int> dims_A(d);
    vector<int> dims_B(d);
    
    // Fill dims_A and dims_B with appropriate dimensions
    for(int i=0; i<d; i++) {
        if(i < d-A->shape.size()) {
            dims_A[i] = 1;
        } else {
            dims_A[i] = A->shape[i-(d-A->shape.size())];
        }
        
        if(i < d-B->shape.size()) {
            dims_B[i] = 1;
        } else {
            dims_B[i] = B->shape[i-(d-B->shape.size())];
        }
    }
    
    subtract(A, B, result, dims_A, dims_B, vector<int>(), 0);

    if(A->requires_grad || B->requires_grad){
        result->backward_fn = [A, B, result](){
            if(A->requires_grad && A->grad!=nullptr){
                auto reduced_grad = result->grad->reduce_to_shape(A->shape);
                reduced_grad->requires_grad = false;
                A->grad = A->grad + reduced_grad;
            }
            if(B->requires_grad && B->grad!=nullptr){
                auto reduced_grad = result->grad->reduce_to_shape(B->shape);
                reduced_grad->requires_grad = false;
                B->grad = B->grad - reduced_grad;
            }
        };
    }
    return result;
}

shared_ptr<Tensor>& operator+=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    auto result = A + B;
    A->data = result->data;
    A->shape = result->shape;
    A->strides = result->strides;
    A->grad = result->grad;
    A->backward_fn = result->backward_fn;
    return A;
}

shared_ptr<Tensor>& operator-=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    auto result = A - B;
    A->data = result->data;
    A->shape = result->shape;
    A->strides = result->strides;
    A->grad = result->grad;
    A->backward_fn = result->backward_fn;
    return A;
}

void Tensor::print() {
    // Helper function to print a single value with proper formatting
    auto print_value = [](float val) {
        cout << val;
        if (val >= 0) cout << " ";  // Add space for positive numbers for better alignment
    };

    // Helper function to print a single dimension
    function<void(vector<int>&, int)> print_dim = [&](vector<int>& indices, int dim) {
        if (dim == shape.size()) {
            // Print the value at these indices
            print_value(at(indices));
            return;
        }

        // Print opening bracket for this dimension
        if (dim > 0) cout << "[";
        
        // Print all elements in this dimension
        for (int i = 0; i < shape[dim]; i++) {
            indices[dim] = i;
            
            // If this is the last dimension, print values directly
            if (dim == shape.size() - 1) {
                print_value(at(indices));
                if (i < shape[dim] - 1) cout << ", ";
            } else {
                // For higher dimensions, print with proper nesting
                print_dim(indices, dim + 1);
                if (i < shape[dim] - 1) cout << ", ";
            }
        }
        
        // Print closing bracket for this dimension
        if (dim > 0) cout << "]";
        
        // Add newline after each complete element of the outer dimension
        if (dim == 0 && indices[0] < shape[0] - 1) cout << endl;
    };

    // Initialize indices vector
    vector<int> indices(shape.size(), 0);
    
    // Print the tensor
    cout << "Tensor(";
    for (int i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i < shape.size() - 1) cout << ", ";
    }
    cout << "):" << endl;
    
    print_dim(indices, 0);
    cout << endl;
}