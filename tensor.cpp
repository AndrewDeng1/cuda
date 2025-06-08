#include "tensor.h"

Tensor::Tensor() : data(nullptr), grad(nullptr), parents(), backward_fn() {}

Tensor::Tensor(const vector<int>& shape, bool requires_grad) : shape(shape), requires_grad(requires_grad) {
    
    this->requires_grad = requires_grad;
    this->shape = shape;
    
    this->strides = compute_strides(shape);

    this->data = new float[size()]();  // Changed from int to float

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

Tensor::Tensor(const vector<int>& shape, float* data, bool requires_grad) : Tensor(shape, requires_grad) {
    for(int i=0; i<size(); i++){
        this->data[i] = data[i];
    }
}

Tensor::Tensor(shared_ptr<Tensor> other) : Tensor(other->shape, other->data, other->requires_grad) {

    // if(requires_grad){
    //     this->grad = make_shared<Tensor>(other->grad);
    // } else {
    //     this->grad = nullptr;
    // }

    // NOTICE HOW COPY CONSTRUCTOR DOESN'T COPY PARENTS AND BACKWARD FN
    // ** ON PURPOSE, MAYBE CHANGE IN FUTURE, BUT PROB NOT **
    // this->parents = other->parents;
    // this->backward_fn = other->backward_fn;
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

float& Tensor::at(int index) {
    if(index < 0 || index >= size()) {
        throw std::runtime_error("Index out of bounds");
    }
    return data[index];
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

    int cnt = 1;
    for (int dim : new_shape) cnt *= dim;
    
    if(cnt!=size()){
        throw std::runtime_error("Reshape size mismatch");
    }

    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, data, requires_grad);

    if(requires_grad) {
        result->parents.push_back(shared_from_this());
        result->backward_fn = [this, result]() {
            // Reshape the gradient back to the original shape
            auto reshaped_grad = result->grad->reshape(shape);
            reshaped_grad->requires_grad = false;
            this->grad = this->grad + reshaped_grad;
        };
    }

    return result;
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

// Assumes keepdims == true
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

    for(int i=0; i<result->size(); i++){
        for(int j=0; j<shape[axis]; j++){
            result->at(i)+=at(i+j*strides[axis]);
        }
    }

    if(requires_grad){
        result->parents.push_back(shared_from_this());
        result->backward_fn = [this, result]() {
            this->grad = this->grad + result->grad->broadcast(shape);
        };
    }

    return result;
    
    // // Anonymous lambda function
    // // Must capture itself, since needs to call itself recursively
    // function<void(shared_ptr<Tensor>, shared_ptr<Tensor>, vector<int>, int, int)> rec = 
    //     [&](shared_ptr<Tensor> A, shared_ptr<Tensor> result, vector<int> acc, int axis, int ind) {
    //         if(ind==result->shape.size()){
    //             float val=0;
    //             vector<int> acc_A(acc);
    //             acc_A[axis]=0;
    //             for(int i=0; i<A->shape[axis]; i++){
    //                 val+=A->at(acc_A);
    //                 acc_A[axis]++;
    //             }
    //             result->at(acc)=val;
    //             return;
    //         }

    //         for(int i=0; i<result->shape[ind]; i++){
    //             vector<int> new_acc(acc);
    //             new_acc.push_back(i);
    //             rec(A, result, new_acc, axis, ind+1);
    //         }
    //     };

    // rec(shared_from_this(), result, vector<int>(), axis, 0);
    // return result;

    // NO BACKWARD FUNCTION IMPLEMENTED RN
}

bool is_broadcastable(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul) {
    int d = max(A_shape.size(), B_shape.size());
    for(int i = 0; i < d-2*matmul; i++) {
        int dim_A = (i < d-A_shape.size()) ? 1 : A_shape[i-(d-A_shape.size())];
        int dim_B = (i < d-B_shape.size()) ? 1 : B_shape[i-(d-B_shape.size())];
        
        if(dim_A == 1 || dim_B == 1) {
            continue;
        } else if(dim_A != dim_B) {
            return false;
        }
    }
    return true;
}

// ** IMPORTANT **
// ** IF MAT_MUL IS TRUE, YOU ARE EXPECTED TO ADD THE LAST TWO DIMENSIONS MANUALLY, SINCE THEY'RE DIFFERENT **
vector<int> get_broadcast_shape(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul) {
    if(!is_broadcastable(A_shape, B_shape, matmul)) {
        throw std::runtime_error("Broadcast shape mismatch");
    }

    vector<int> new_shape;
    int d = max(A_shape.size(), B_shape.size());
    for(int i=0; i<d-2*matmul; i++){
        if(i<d-A_shape.size()){
            new_shape.push_back(B_shape[i]);
        } else if(i<d-B_shape.size()){
            new_shape.push_back(A_shape[i]);
        } else {
            if(A_shape[i-(d-A_shape.size())]==1){
                new_shape.push_back(B_shape[i-(d-B_shape.size())]);
            } else if(B_shape[i-(d-B_shape.size())]==1){
                new_shape.push_back(A_shape[i-(d-A_shape.size())]);
            } else if(A_shape[i-(d-A_shape.size())]==B_shape[i-(d-B_shape.size())]){
                new_shape.push_back(A_shape[i-(d-A_shape.size())]);
            } else {
                throw invalid_argument("Shape mismatch");
            }
        }
    }

    return new_shape;
}

shared_ptr<Tensor> Tensor::broadcast(const vector<int>& new_shape, bool matmul) {
    if(!is_broadcastable(shape, new_shape, matmul)) {
        throw std::runtime_error("Broadcast shape mismatch");
    }

    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, requires_grad);

    // Create padded shape and strides
    vector<int> padded_shape = shape;
    vector<int> padded_strides = strides;
    
    // Pad with 1s at the beginning if needed
    while(padded_shape.size() < new_shape.size()) {
        padded_shape.insert(padded_shape.begin(), 1);
        padded_strides.insert(padded_strides.begin(), 0);  // Stride of 0 for size-1 dimensions
    }

    for(int i=0; i<result->size(); i++) {
        int curr = i;
        int idx = 0;
        for(int j=0; j<new_shape.size()-2*matmul; j++) {
            int dim = curr/result->strides[j];
            curr %= result->strides[j];
            
            if(padded_shape[j] == 1) {
                idx += 0;  // Don't add to index for broadcasted dimensions
            } else {
                idx += padded_strides[j] * dim;
            }
        }
        result->at(i) = at(idx);
    }

    if(requires_grad) {
        result->parents.push_back(shared_from_this());
        result->backward_fn = [this, result]() {
            // Sum gradients across broadcasted dimensions
            auto reduced_grad = result->grad->reduce_to_shape(shape);
            reduced_grad->requires_grad = false;
            this->grad = this->grad + reduced_grad;
        };
    }

    return result;
}

shared_ptr<Tensor> Tensor::reduce_to_shape(const vector<int>& target_shape) {

    if(target_shape.size() > shape.size()){
        throw std::runtime_error("Target shape size mismatch");
    }

    if(target_shape.size() == shape.size()){
        bool same = true;
        for(int i=0; i<target_shape.size(); i++){
            if(target_shape[i]!=shape[i]){
                same = false;
                break;
            }
        }
        if(same){
            return shared_from_this();
        }
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

    shared_ptr<Tensor> result = shared_from_this();
    for(int i=0; i<new_shape.size(); i++) {
        if(new_shape[i]==1 && shape[i]!=1) {
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

    if(!is_broadcastable(A->shape, B->shape, false)){
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, false);

    shared_ptr<Tensor> new_A = A->broadcast(new_shape, false);
    shared_ptr<Tensor> new_B = B->broadcast(new_shape, false);

    if(new_A->size()!=new_B->size()){
        throw std::runtime_error("Broadcast size mismatch");
    }

    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad||B->requires_grad);

    for(int i=0; i<new_A->size(); i++){
        result->at(i) = new_A->at(i) + new_B->at(i);
    }

    if(A->requires_grad || B->requires_grad){

        // ** If parent's don't require grad, then don't add them?
        result->parents.push_back(A);
        result->parents.push_back(B);

        result->backward_fn = [A, B, result](){
            if(A->requires_grad && A->grad!=nullptr){
                A->grad += result->grad->reduce_to_shape(A->shape);
            }
            if(B->requires_grad && B->grad!=nullptr){
                B->grad += result->grad->reduce_to_shape(B->shape);
            }
        };
    }

    return result;



    // vector<int> new_shape;
    // int d = max(A->shape.size(), B->shape.size());
    // for(int i=0; i<d; i++){
    //     if(i<d-A->shape.size()){
    //         new_shape.push_back(B->shape[i]);
    //     } else if(i<d-B->shape.size()){
    //         new_shape.push_back(A->shape[i]);
    //     } else {
    //         if(A->shape[i-(d-A->shape.size())]==1){
    //             new_shape.push_back(B->shape[i-(d-B->shape.size())]);
    //         } else if(B->shape[i-(d-B->shape.size())]==1){
    //             new_shape.push_back(A->shape[i-(d-A->shape.size())]);
    //         } else if(A->shape[i-(d-A->shape.size())]==B->shape[i-(d-B->shape.size())]){
    //             new_shape.push_back(A->shape[i-(d-A->shape.size())]);
    //         } else {
    //             throw invalid_argument("Shape mismatch");
    //         }
    //     }
    // }

    // shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad||B->requires_grad);
    
    // // Convert add to a lambda function
    // function<void(shared_ptr<Tensor>, shared_ptr<Tensor>, shared_ptr<Tensor>, vector<int>&, vector<int>&, vector<int>, int)> add = 
    //     [&](shared_ptr<Tensor> A, shared_ptr<Tensor> B, shared_ptr<Tensor> C, vector<int>& dims_A, vector<int>& dims_B, vector<int> acc, int ind) {
    //         if(ind==dims_A.size()){
    //             vector<int> acc_A(acc);
    //             for(int i=0; i<acc_A.size(); i++){
    //                 if(dims_A[i]==1){
    //                     acc_A[i]=0;
    //                 }
    //             }
    //             vector<int> acc_B(acc);
    //             for(int i=0; i<acc_B.size(); i++){
    //                 if(dims_B[i]==1){
    //                     acc_B[i]=0;
    //                 }
    //             }
    //             C->at(acc)=A->at(acc_A)+B->at(acc_B);
    //             return;
    //         }

    //         for(int i=0; i<max(dims_A[ind], dims_B[ind]); i++){
    //             vector<int> new_acc(acc);
    //             new_acc.push_back(i);
    //             add(A, B, C, dims_A, dims_B, new_acc, ind+1);
    //         }
    //     };

    // vector<int> dims_A(d);
    // vector<int> dims_B(d);
    
    // // Fill dims_A and dims_B with appropriate dimensions
    // for(int i=0; i<d; i++) {
    //     if(i < d-A->shape.size()) {
    //         dims_A[i] = 1;
    //     } else {
    //         dims_A[i] = A->shape[i-(d-A->shape.size())];
    //     }
        
    //     if(i < d-B->shape.size()) {
    //         dims_B[i] = 1;
    //     } else {
    //         dims_B[i] = B->shape[i-(d-B->shape.size())];
    //     }
    // }
    
    // add(A, B, result, dims_A, dims_B, vector<int>(), 0);
}

shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    if(!is_broadcastable(A->shape, B->shape, false)){
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, false);

    shared_ptr<Tensor> new_A = A->broadcast(new_shape, false);
    shared_ptr<Tensor> new_B = B->broadcast(new_shape, false);

    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad||B->requires_grad);

    for(int i=0; i<new_A->size(); i++){
        result->at(i) = new_A->at(i) - new_B->at(i);
    }

    if(A->requires_grad || B->requires_grad){

        // ** If parent's don't require grad, then don't add them?
        result->parents.push_back(A);
        result->parents.push_back(B);

        result->backward_fn = [A, B, result](){
            if(A->requires_grad && A->grad!=nullptr){
                A->grad += result->grad->reduce_to_shape(A->shape);
            }
            if(B->requires_grad && B->grad!=nullptr){
                B->grad += result->grad->reduce_to_shape(B->shape);
            }
        };
    }
    
    return result;
    // vector<int> new_shape;
    // int d = max(A->shape.size(), B->shape.size());
    // for(int i=0; i<d; i++){
    //     if(i<d-A->shape.size()){
    //         new_shape.push_back(B->shape[i]);
    //     } else if(i<d-B->shape.size()){
    //         new_shape.push_back(A->shape[i]);
    //     } else {
    //         if(A->shape[i-(d-A->shape.size())]==1){
    //             new_shape.push_back(B->shape[i-(d-B->shape.size())]);
    //         } else if(B->shape[i-(d-B->shape.size())]==1){
    //             new_shape.push_back(A->shape[i-(d-A->shape.size())]);
    //         } else if(A->shape[i-(d-A->shape.size())]==B->shape[i-(d-B->shape.size())]){
    //             new_shape.push_back(A->shape[i-(d-A->shape.size())]);
    //         } else {
    //             throw invalid_argument("Shape mismatch");
    //         }
    //     }
    // }

    // shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad||B->requires_grad);
    
    // // Convert subtract to a lambda function
    // function<void(shared_ptr<Tensor>, shared_ptr<Tensor>, shared_ptr<Tensor>, vector<int>&, vector<int>&, vector<int>, int)> subtract = 
    //     [&](shared_ptr<Tensor> A, shared_ptr<Tensor> B, shared_ptr<Tensor> C, vector<int>& dims_A, vector<int>& dims_B, vector<int> acc, int ind) {
    //         if(ind==dims_A.size()){
    //             vector<int> acc_A(acc);
    //             for(int i=0; i<acc_A.size(); i++){
    //                 if(dims_A[i]==1){
    //                     acc_A[i]=0;
    //                 }
    //             }
    //             vector<int> acc_B(acc);
    //             for(int i=0; i<acc_B.size(); i++){
    //                 if(dims_B[i]==1){
    //                     acc_B[i]=0;
    //                 }
    //             }
    //             C->at(acc)=A->at(acc_A)-B->at(acc_B);
    //             return;
    //         }

    //         for(int i=0; i<max(dims_A[ind], dims_B[ind]); i++){
    //             vector<int> new_acc(acc);
    //             new_acc.push_back(i);
    //             subtract(A, B, C, dims_A, dims_B, new_acc, ind+1);
    //         }
    //     };

    // vector<int> dims_A(d);
    // vector<int> dims_B(d);
    
    // // Fill dims_A and dims_B with appropriate dimensions
    // for(int i=0; i<d; i++) {
    //     if(i < d-A->shape.size()) {
    //         dims_A[i] = 1;
    //     } else {
    //         dims_A[i] = A->shape[i-(d-A->shape.size())];
    //     }
        
    //     if(i < d-B->shape.size()) {
    //         dims_B[i] = 1;
    //     } else {
    //         dims_B[i] = B->shape[i-(d-B->shape.size())];
    //     }
    // }
    
    // subtract(A, B, result, dims_A, dims_B, vector<int>(), 0);

    // if(A->requires_grad || B->requires_grad){

    //     result->parents.push_back(A);
    //     result->parents.push_back(B);

    //     result->backward_fn = [A, B, result](){
    //         if(A->requires_grad && A->grad!=nullptr){
    //             auto reduced_grad = result->grad->reduce_to_shape(A->shape);
    //             reduced_grad->requires_grad = false;
    //             A->grad = A->grad + reduced_grad;
    //         }
    //         if(B->requires_grad && B->grad!=nullptr){
    //             auto reduced_grad = result->grad->reduce_to_shape(B->shape);
    //             reduced_grad->requires_grad = false;
    //             B->grad = B->grad - reduced_grad;
    //         }
    //     };
    // }
    // return result;
}

// shared_ptr<Tensor> operator*(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    
//     if(A->shape.size()!=B->shape.size()){
//         throw invalid_argument("Shape mismatch");
//     }

//     for(int i=0; i<A->shape.size(); i++){
//         if(A->shape[i]!=B->shape[i]){
    
// }

shared_ptr<Tensor>& operator+=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {

    // Manages memory and references carefully
    A = A + B;
    return A;
}

shared_ptr<Tensor>& operator-=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    
    // Manages memory and references carefully
    A = A - B;
    return A;
}

shared_ptr<Tensor> Tensor::transpose(int dim1, int dim2) {
    
    // Must be at least 2 dimensions for transpose
    if(shape.size()<2){
        throw std::runtime_error("Tensor must have at least 2 dimensions");
    }

    // Check bounds
    if((dim1>0&&dim1>=shape.size())||
    (dim1<0&&dim1<-shape.size())||
    (dim2>0&&dim2>=shape.size())||
    (dim2<0&&dim2<-shape.size())){
        throw std::runtime_error("Invalid dimension");
    }

    // If equal, then same as original tensor
    if(dim1==dim2){
        return shared_from_this();
    }

    // Convert negative indices to positive
    if(dim1<0){
        dim1+=shape.size();
    }
    if(dim2<0){
        dim2+=shape.size();
    }

    vector<int> new_shape(shape);
    swap(new_shape[dim1], new_shape[dim2]);

    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, data, requires_grad);

    if(requires_grad){

        result->parents.push_back(shared_from_this());

        result->backward_fn = [dim1, dim2, result, this](){
            this->grad += result->grad->transpose(dim1, dim2);
        };
    }

    return result;
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