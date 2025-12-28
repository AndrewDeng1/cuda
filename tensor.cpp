#include "tensor.h"
#include "tensor_kernels.h"
#include <cfloat>
#include <random>

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

Tensor::Tensor(const vector<int>& shape, float num, bool requires_grad) : Tensor(shape, requires_grad) {
    for(int i=0; i<size(); i++){
        this->data[i] = num;
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
            // printf("Backward reshape\n");
            // Reshape the gradient back to the original shape
            auto reshaped_grad = result->grad->reshape(shape);
            reshaped_grad->requires_grad = false;
            this->grad = this->grad + reshaped_grad;
        };
    }

    return result;
}

void Tensor::backward() {
    // Topological sort using Kahn's algorithm
    unordered_map<shared_ptr<Tensor>, int> in_degree;
    queue<shared_ptr<Tensor>> q;
    unordered_set<shared_ptr<Tensor>> visited;
    
    // First, traverse the entire graph to count in-degrees for all tensors
    function<void(shared_ptr<Tensor>)> count_in_degrees = [&](shared_ptr<Tensor> tensor) {
        if (visited.count(tensor)) return;
        visited.insert(tensor);
        
        in_degree[tensor]=0;

        // Count incoming edges for this tensor's parents
        for (const auto& parent : tensor->parents) {
            count_in_degrees(parent);  // Recursively process parents
            in_degree[parent]++;
        }
    };
    
    // Start counting from this tensor
    count_in_degrees(shared_from_this());
    
    // Add tensors with no incoming edges to queue
    for (const auto& pair : in_degree) {
        if (pair.second == 0) {
            q.push(pair.first);
        }
    }

    // Process tensors in topological order
    while (!q.empty()) {
        auto tensor = q.front();
        q.pop();

        // printf("Backpropagating tensor: ");
        // tensor->grad->print();
        // printf("Parents: ");
        // for(const auto& parent : tensor->parents){
        //     printf("(");
        //     parent->grad->print();
        //     printf(")");
        // }
        // printf("\n");
        // printf("--------------------------------\n");
        
        // Call backward function if it exists
        if (tensor->backward_fn) {
            tensor->backward_fn();
        }
        
        // Update in-degree for parents
        for (const auto& parent : tensor->parents) {
            in_degree[parent]--;
            if (in_degree[parent] == 0) {
                q.push(parent);
            }
        }
    }
}

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

    // CPU:
    // for(int i=0; i<result->size(); i++){
    //     for(int j=0; j<shape[axis]; j++){

    //         int curr=i;
    //         int idx=0;
    //         for(int x=0; x<shape.size(); x++){
    //             if(x==axis){
    //                 idx+=j*strides[x];
    //             } else {
    //                 idx+=(curr/result->strides[x])*strides[x];
    //             }
    //             curr%=result->strides[x];
    //         }

    //         result->at(i)+=at(idx);
    //     }
    // }

    launch_sum(shared_from_this(), result, axis);

    if(requires_grad){
        result->parents.push_back(shared_from_this());
        result->backward_fn = [this, result]() {
            // printf("Backward sum\n");
            this->grad = this->grad + result->grad->broadcast(shape);
        };
    }

    return result;
}

bool is_broadcastable(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul) {
    
    if(matmul && (A_shape.size()<2 || B_shape.size()<2)){
        return false;
    }

    if(matmul && A_shape[A_shape.size()-1]!=B_shape[B_shape.size()-2]){
        return false;
    }
    
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
    // Checks matmul compatibility
    if(!is_broadcastable(shape, new_shape, matmul)) {
        throw std::runtime_error("Broadcast shape mismatch");
    }

    bool same_shape=true;
    for(int i=0; i<std::min(shape.size(), new_shape.size()); i++){
        if(shape[i]!=new_shape[i]){
            same_shape=false;
            break;
        }
    }
    if(shape.size()!=new_shape.size())same_shape=false;

    if(same_shape){
        return shared_from_this();
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

    // CPU:
    // for(int i=0; i<result->size(); i++) {
    //     int curr = i;
    //     int idx = 0;
    //     for(int j=0; j<new_shape.size()-2*matmul; j++) {
    //         int dim = curr/result->strides[j];
    //         curr %= result->strides[j];
            
    //         if(padded_shape[j] == 1) {
    //             idx += 0;  // Don't add to index for broadcasted dimensions
    //         } else {
    //             idx += padded_strides[j] * dim;
    //         }
    //     }
    //     result->at(i) = at(idx);
    // }

    launch_broadcast(shared_from_this(), result, padded_shape, padded_strides, matmul);

    if(requires_grad) {
        result->parents.push_back(shared_from_this());
        result->backward_fn = [this, result]() {
            // printf("Backward broadcast\n");
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

    // CPU:
    // for(int i=0; i<new_A->size(); i++){
    //     result->at(i) = new_A->at(i) + new_B->at(i);
    // }

    launch_add(new_A, new_B, result);

    if(new_A->requires_grad || new_B->requires_grad){
        result->parents.push_back(new_A);
        result->parents.push_back(new_B);

        result->backward_fn = [new_A, new_B, result](){
            // printf("Backward +\n");

            // TODO: Confirm that this doesn't result in infinite loop
            if(new_A->requires_grad && new_A->grad!=nullptr){
                new_A->grad += result->grad->reduce_to_shape(new_A->shape);
            }
            if(new_B->requires_grad && new_B->grad!=nullptr){
                new_B->grad += result->grad->reduce_to_shape(new_B->shape);
            }
        };
    }

    return result;
}

shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    if(!is_broadcastable(A->shape, B->shape, false)){
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, false);

    shared_ptr<Tensor> new_A = A->broadcast(new_shape, false);
    shared_ptr<Tensor> new_B = B->broadcast(new_shape, false);

    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad||B->requires_grad);

    // CPU:
    // for(int i=0; i<new_A->size(); i++){
    //     result->at(i) = new_A->at(i) - new_B->at(i);
    // }

    launch_subtract(new_A, new_B, result);

    if(new_A->requires_grad || new_B->requires_grad){
        result->parents.push_back(new_A);
        result->parents.push_back(new_B);

        result->backward_fn = [new_A, new_B, result](){
            // printf("Backward -\n");
            if(new_A->requires_grad && new_A->grad!=nullptr){
                new_A->grad += result->grad->reduce_to_shape(new_A->shape);
            }
            if(new_B->requires_grad && new_B->grad!=nullptr){
                new_B->grad -= result->grad->reduce_to_shape(new_B->shape);
            }
        };
    }
    
    return result;
}

shared_ptr<Tensor> operator*(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    
    if(!is_broadcastable(A->shape, B->shape, false)){
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, false);

    shared_ptr<Tensor> new_A = A->broadcast(new_shape, false);
    shared_ptr<Tensor> new_B = B->broadcast(new_shape, false);
    
    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad||B->requires_grad);

    // CPU:
    // for(int i=0; i<new_A->size(); i++){
    //     result->at(i) = new_A->at(i) * new_B->at(i);
    // }

    launch_multiply(new_A, new_B, result);

    if(new_A->requires_grad || new_B->requires_grad){
        result->parents.push_back(new_A);
        result->parents.push_back(new_B);

        result->backward_fn = [new_A, new_B, result](){
            // printf("Backward *\n");
            if(new_A->requires_grad && new_A->grad!=nullptr){
                new_A->grad += result->grad * new_B;
            }
            if(new_B->requires_grad && new_B->grad!=nullptr){
                new_B->grad += result->grad * new_A;
            }
        };
    }

    return result;
}

shared_ptr<Tensor> operator/(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    if(!is_broadcastable(A->shape, B->shape, false)){
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, false);

    shared_ptr<Tensor> new_A = A->broadcast(new_shape, false);
    shared_ptr<Tensor> new_B = B->broadcast(new_shape, false);
    
    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad||B->requires_grad);

    // CPU:
    // for(int i=0; i<new_A->size(); i++){
    //     result->at(i) = new_A->at(i) / new_B->at(i);
    // }

    launch_divide(new_A, new_B, result);

    if(new_A->requires_grad || new_B->requires_grad){
        result->parents.push_back(new_A);
        result->parents.push_back(new_B);
        
        result->backward_fn = [new_A, new_B, result](){
            // printf("Backward /\n");
            if(new_A->requires_grad){
                new_A->grad += result->grad/new_B;
            }
            if(new_B->requires_grad){
                new_B->grad += result->grad*(-new_A/(new_B*new_B));
            }
        };
    }

    return result;
}
    
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

shared_ptr<Tensor>& operator*=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    A = A * B;
    return A;
}

shared_ptr<Tensor>& operator/=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    A = A / B;
    return A;
}

shared_ptr<Tensor> operator+(const shared_ptr<Tensor>& A, float B) {
    return A + make_shared<Tensor>(vector<int>{1}, vector<float>{B}, true);
}

shared_ptr<Tensor> operator+(float A, const shared_ptr<Tensor>& B) {
    return make_shared<Tensor>(vector<int>{1}, vector<float>{A}, true) + B;
}

shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A, float B) {
    return A - make_shared<Tensor>(vector<int>{1}, vector<float>{B}, true);
}

shared_ptr<Tensor> operator-(float A, const shared_ptr<Tensor>& B) {
    return make_shared<Tensor>(vector<int>{1}, vector<float>{A}, true) - B;
}

shared_ptr<Tensor> operator*(const shared_ptr<Tensor>& A, float B) {
    return A * make_shared<Tensor>(vector<int>{1}, vector<float>{B}, true);
}

shared_ptr<Tensor> operator*(float A, const shared_ptr<Tensor>& B) {
    return make_shared<Tensor>(vector<int>{1}, vector<float>{A}, true) * B;
}

shared_ptr<Tensor> operator/(const shared_ptr<Tensor>& A, float B) {
    return A / make_shared<Tensor>(vector<int>{1}, vector<float>{B}, true);
}

shared_ptr<Tensor> operator/(float A, const shared_ptr<Tensor>& B) {
    return make_shared<Tensor>(vector<int>{1}, vector<float>{A}, true) / B;
}

shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A) {
    return -1.0f * A;
}

// // Comparison operators
// shared_ptr<Tensor> operator>(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
//     if(!is_broadcastable(A->shape, B->shape, false)){
//         throw invalid_argument("Shape mismatch");
//     }

//     vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, false);

//     shared_ptr<Tensor> new_A = A->broadcast(new_shape, false);
//     shared_ptr<Tensor> new_B = B->broadcast(new_shape, false);
    
//     shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad || B->requires_grad);

//     for(int i=0; i<new_A->size(); i++){
//         result->at(i) = new_A->at(i) > new_B->at(i) ? 1.0f : 0.0f;
//     }

//     if(new_A->requires_grad || new_B->requires_grad){
//         result->parents.push_back(new_A);
//         result->parents.push_back(new_B);
//         result->backward_fn = [](){
//             throw runtime_error("Backward function for comparison operator '>' should not be called directly");
//         };
//     }

//     return result;
// }

// shared_ptr<Tensor> operator<(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
//     if(!is_broadcastable(A->shape, B->shape, false)){
//         throw invalid_argument("Shape mismatch");
//     }

//     vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, false);

//     shared_ptr<Tensor> new_A = A->broadcast(new_shape, false);
//     shared_ptr<Tensor> new_B = B->broadcast(new_shape, false);
    
//     shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad || B->requires_grad);

//     for(int i=0; i<new_A->size(); i++){
//         result->at(i) = new_A->at(i) < new_B->at(i) ? 1.0f : 0.0f;
//     }

//     if(new_A->requires_grad || new_B->requires_grad){
//         result->parents.push_back(new_A);
//         result->parents.push_back(new_B);
//         result->backward_fn = [](){
//             throw runtime_error("Backward function for comparison operator '<' should not be called directly");
//         };
//     }

//     return result;
// }

// shared_ptr<Tensor> operator>=(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
//     if(!is_broadcastable(A->shape, B->shape, false)){
//         throw invalid_argument("Shape mismatch");
//     }

//     vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, false);

//     shared_ptr<Tensor> new_A = A->broadcast(new_shape, false);
//     shared_ptr<Tensor> new_B = B->broadcast(new_shape, false);
    
//     shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad || B->requires_grad);

//     for(int i=0; i<new_A->size(); i++){
//         result->at(i) = new_A->at(i) >= new_B->at(i) ? 1.0f : 0.0f;
//     }

//     if(new_A->requires_grad || new_B->requires_grad){
//         result->parents.push_back(new_A);
//         result->parents.push_back(new_B);
//         result->backward_fn = [](){
//             throw runtime_error("Backward function for comparison operator '>=' should not be called directly");
//         };
//     }

//     return result;
// }

// shared_ptr<Tensor> operator<=(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
//     if(!is_broadcastable(A->shape, B->shape, false)){
//         throw invalid_argument("Shape mismatch");
//     }

//     vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, false);

//     shared_ptr<Tensor> new_A = A->broadcast(new_shape, false);
//     shared_ptr<Tensor> new_B = B->broadcast(new_shape, false);
    
//     shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, A->requires_grad || B->requires_grad);

//     for(int i=0; i<new_A->size(); i++){
//         result->at(i) = new_A->at(i) <= new_B->at(i) ? 1.0f : 0.0f;
//     }

//     if(new_A->requires_grad || new_B->requires_grad){
//         result->parents.push_back(new_A);
//         result->parents.push_back(new_B);
//         result->backward_fn = [](){
//             throw runtime_error("Backward function for comparison operator '<=' should not be called directly");
//         };
//     }

//     return result;
// }

// // Scalar comparison operators
// shared_ptr<Tensor> operator>(const shared_ptr<Tensor>& A, float B) {
//     return A > make_shared<Tensor>(vector<int>{1}, vector<float>{B}, A->requires_grad);
// }

// shared_ptr<Tensor> operator>(float A, const shared_ptr<Tensor>& B) {
//     return make_shared<Tensor>(vector<int>{1}, vector<float>{A}, B->requires_grad) > B;
// }

// shared_ptr<Tensor> operator<(const shared_ptr<Tensor>& A, float B) {
//     return A < make_shared<Tensor>(vector<int>{1}, vector<float>{B}, A->requires_grad);
// }

// shared_ptr<Tensor> operator<(float A, const shared_ptr<Tensor>& B) {
//     return make_shared<Tensor>(vector<int>{1}, vector<float>{A}, B->requires_grad) < B;
// }

// shared_ptr<Tensor> operator>=(const shared_ptr<Tensor>& A, float B) {
//     return A >= make_shared<Tensor>(vector<int>{1}, vector<float>{B}, A->requires_grad);
// }

// shared_ptr<Tensor> operator>=(float A, const shared_ptr<Tensor>& B) {
//     return make_shared<Tensor>(vector<int>{1}, vector<float>{A}, B->requires_grad) >= B;
// }

// shared_ptr<Tensor> operator<=(const shared_ptr<Tensor>& A, float B) {
//     return A <= make_shared<Tensor>(vector<int>{1}, vector<float>{B}, A->requires_grad);
// }

// shared_ptr<Tensor> operator<=(float A, const shared_ptr<Tensor>& B) {
//     return make_shared<Tensor>(vector<int>{1}, vector<float>{A}, B->requires_grad) <= B;
// }

shared_ptr<Tensor> matmul(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B) {
    
    // is_broadcast should alr check this, but check anyways
    if(A->shape.size()<2 || B->shape.size()<2){
        throw invalid_argument("Tensor must have at least 2 dimensions");
    }

    if(A->shape[A->shape.size()-1]!=B->shape[B->shape.size()-2]){
        throw invalid_argument("Shape mismatch");
    }

    if(!is_broadcastable(A->shape, B->shape, true)){
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A->shape, B->shape, true);

    vector<int> new_shape_A(new_shape);
    new_shape_A.push_back(A->shape[A->shape.size()-2]);
    new_shape_A.push_back(A->shape[A->shape.size()-1]);
    vector<int> new_shape_B(new_shape);
    new_shape_B.push_back(B->shape[B->shape.size()-2]);
    new_shape_B.push_back(B->shape[B->shape.size()-1]);

    shared_ptr<Tensor> new_A = A->broadcast(new_shape_A);
    shared_ptr<Tensor> new_B = B->broadcast(new_shape_B);
    vector<int> matmul_output_shape;
    for(int i=0; i<new_shape.size(); i++){
        matmul_output_shape.push_back(new_shape[i]);
    }
    matmul_output_shape.push_back(new_shape_A[new_shape_A.size()-2]);
    matmul_output_shape.push_back(new_shape_B[new_shape_B.size()-1]);

    shared_ptr<Tensor> result = make_shared<Tensor>(matmul_output_shape, A->requires_grad||B->requires_grad);
    
    // --- CPU START ---
    // for(int i=0; i<result->size(); i++){
    //     int ind_A = 0;
    //     int ind_B = 0;
    //     int curr_A = i;
    //     int curr_B = i;

    //     // Gets you to dimension before 2D
    //     for(int j=0; j<new_shape_A.size()-2; j++){
    //         ind_A += (curr_A/result->strides[j])*new_A->strides[j];
    //         curr_A%=result->strides[j];
    //     }

    //     // Get corresponding row
    //     ind_A += (curr_A/result->strides[result->strides.size()-2])*new_A->strides[new_A->strides.size()-2];
        
    //     // Ignore column
    //     curr_A%=result->strides[result->strides.size()-1];
        
    //     // Gets you to dimension before 2D
    //     for(int j=0; j<new_shape_B.size()-2; j++){
    //         ind_B += (curr_B/result->strides[j])*new_B->strides[j];
    //         curr_B%=result->strides[j];
    //     }

    //     // Ignore row
    //     curr_B%=result->strides[result->strides.size()-2];
        
    //     // Get corresponding column
    //     ind_B += (curr_B/result->strides[result->strides.size()-1])*new_B->strides[new_B->strides.size()-1];
        
    //     for(int j=0; j<new_shape_A[new_shape_A.size()-1]; j++){
    //         result->at(i) += new_A->at(ind_A)*new_B->at(ind_B);
    //         ind_A += new_A->strides[new_A->strides.size()-1];
    //         ind_B += new_B->strides[new_B->strides.size()-2];
    //     }
    // }
    // --- CPU END ---

    launch_matmul(new_A, new_B, result);
    
    if(new_A->requires_grad || new_B->requires_grad){
        result->parents.push_back(new_A);
        result->parents.push_back(new_B);
        
        result->backward_fn = [new_A, new_B, result]() {
            // printf("Backward matmul\n");
            if(new_A->requires_grad) {
                // dA = dC * B^T
                auto B_transposed = new_B->transpose(new_B->shape.size()-2, new_B->shape.size()-1);
                new_A->grad += matmul(result->grad, B_transposed);
            }
            if(new_B->requires_grad) {
                // dB = A^T * dC
                auto A_transposed = new_A->transpose(new_A->shape.size()-2, new_A->shape.size()-1);
                new_B->grad += matmul(A_transposed, result->grad);
            }
        };
    }

    return result;
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

    // CPU:
    // for(int i=0; i<result->size(); i++){
    //     int ind=0;
    //     int curr=i;
    //     int mag_dim1;
    //     int mag_dim2;
    //     for(int j=0; j<new_shape.size(); j++){
    //         if(j==dim1){
    //             mag_dim1=curr/result->strides[j];
    //         } else if(j==dim2){
    //             mag_dim2=curr/result->strides[j];
    //         }
    //         ind+=(curr/result->strides[j])*strides[j];
    //         curr%=result->strides[j];
    //     }
    //     ind-=mag_dim1*strides[dim1];
    //     ind+=mag_dim2*strides[dim1];
    //     ind+=mag_dim1*strides[dim2];
    //     ind-=mag_dim2*strides[dim2];
    //     result->at(i)=at(ind);
    // }

    launch_transpose(shared_from_this(), result, dim1, dim2);

    if(requires_grad){
        result->parents.push_back(shared_from_this());

        result->backward_fn = [dim1, dim2, result, this](){
            // printf("Backward transpose\n");
            this->grad += result->grad->transpose(dim1, dim2);
        };
    }

    return result;
}

shared_ptr<Tensor> Tensor::pow(float exponent) {
    shared_ptr<Tensor> result = make_shared<Tensor>(shape, requires_grad);
    
    // CPU:
    // for(int i=0; i<result->size(); i++){
    //     result->at(i) = std::pow(at(i), exponent);
    // }

    launch_pow(shared_from_this(), result, exponent);

    if(requires_grad){
        result->parents.push_back(shared_from_this());
        result->backward_fn = [exponent, result, this](){
            this->grad += result->grad*exponent*pow(exponent-1.0f);
        };
    }
    return result;
}

shared_ptr<Tensor> Tensor::mean(int axis, bool keepdims) {
    int N = shape[axis];
    return sum(axis, keepdims)/N;
}

shared_ptr<Tensor> Tensor::variance_squared(int axis, bool keepdims) {
    shared_ptr<Tensor> centered = shared_from_this()-mean(axis, keepdims)->broadcast(shape, false);
    shared_ptr<Tensor> centered_squared = centered*centered;
    return centered_squared->mean(axis, keepdims);
}

shared_ptr<Tensor> relu(const shared_ptr<Tensor>& A) {
    shared_ptr<Tensor> result = make_shared<Tensor>(A->shape, A->requires_grad);
    
    // CPU:
    // for(int i=0; i<result->size(); i++){
    //     result->at(i) = std::max(0.0f, A->at(i));
    // }

    launch_relu(A, result);

    if(A->requires_grad){
        result->parents.push_back(A);
        result->backward_fn = [A, result](){
            // For ReLU, we need to compute the gradient mask without using comparison operators
            // since they don't have backward functions
            auto grad_mask = make_shared<Tensor>(result->shape, false);
            for(int i=0; i<result->size(); i++) {
                grad_mask->at(i) = result->at(i) > 0.0f ? 1.0f : 0.0f;
            }
            A->grad += result->grad * grad_mask;
        };
    }
    return result;
}

shared_ptr<Tensor> sigmoid(const shared_ptr<Tensor>& A) {
    shared_ptr<Tensor> result = make_shared<Tensor>(A->shape, A->requires_grad);
    
    // CPU:
    // for(int i=0; i<result->size(); i++){
    //     result->at(i) = 1.0f/(1.0f+exp(-A->at(i)));
    // }

    launch_sigmoid(A, result);

    if(A->requires_grad){
        result->parents.push_back(A);
        result->backward_fn = [A, result](){
            A->grad += result->grad*result*(1.0f-result);
        };
    }
    return result;
}

shared_ptr<Tensor> tanh(const shared_ptr<Tensor>& A) {
    shared_ptr<Tensor> result = make_shared<Tensor>(A->shape, A->requires_grad);
    
    // CPU:
    // for(int i=0; i<result->size(); i++){
    //     result->at(i) = (exp(A->at(i)) - exp(-A->at(i))) / (exp(A->at(i)) + exp(-A->at(i)));
    // }

    launch_tanh(A, result);

    if(A->requires_grad){
        result->parents.push_back(A);
        result->backward_fn = [A, result](){
            A->grad += result->grad*(1.0f-result*result);
        };
    }
    return result;
}

shared_ptr<Tensor> dropout(const shared_ptr<Tensor>& A, float p, bool training) {
    shared_ptr<Tensor> mask = make_shared<Tensor>(A->shape, false);
    
    // Generate mask on CPU
    if(!training || p == 0.0f) {
        for(int i = 0; i < mask->size(); i++) {
            mask->at(i) = 1.0f;
        }
    } else {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - p);
        float scale = 1.0f / (1.0f - p);
        
        for(int i = 0; i < mask->size(); i++) {
            mask->at(i) = dist(gen) ? scale : 0.0f;
        }
    }

    shared_ptr<Tensor> result = A * mask;
    
    if(A->requires_grad) {
        result->parents.push_back(A);
        result->backward_fn = [A, result, mask]() {
            A->grad += result->grad * mask;
        };
    }
    return result;
}

// SOFTMAX
shared_ptr<Tensor> Tensor::softmax(int axis) {
    if(axis<0){
        axis+=shape.size();
    }
    if(axis>=shape.size()||axis<0){
        throw std::runtime_error("Invalid axis");
    }

    // Equivalent of sum along axis with keepdims=true, but sum of exp(at(i))
    vector<int> sm_exp_shape;
    for(int i=0; i<shape.size(); i++){
        if(i==axis){
            sm_exp_shape.push_back(1);
        } else {
            sm_exp_shape.push_back(shape[i]);
        }
    }

    shared_ptr<Tensor> sm_exp = make_shared<Tensor>(sm_exp_shape, requires_grad);

    // CPU START:
    // // TODO: Same with cuda version, when dtype becomes adjustable, will need diff way to get max value for dtype
    // // TODO: Same with cuda version, should be getting max along axis and not across entire tensor, and then shifting based on max along each axis
    // float mx=-FLT_MAX;
    // for(int i=0; i<sm_exp->size(); i++){
    //     mx=std::max(mx, at(i));
    // }

    // for(int i=0; i<sm_exp->size(); i++){
    //     for(int j=0; j<shape[axis]; j++){

    //         int curr=i;
    //         int idx=0;
    //         for(int x=0; x<shape.size(); x++){
    //             if(x==axis){
    //                 idx+=j*strides[x];
    //             } else {
    //                 idx+=(curr/sm_exp->strides[x])*strides[x];
    //             }
    //             curr%=sm_exp->strides[x];
    //         }

    //         sm_exp->at(i)+=exp(at(idx)-mx);
    //     }
    // }
    

    // shared_ptr<Tensor> sm_exp_broadcast = sm_exp->broadcast(shape, false);

    // // Softmax should return the same shape as input
    // shared_ptr<Tensor> result = make_shared<Tensor>(shape, requires_grad);

    // for(int i=0; i<result->size(); i++){
    //     result->at(i)=exp(at(i)-mx)/sm_exp_broadcast->at(i);
    // }
    // CPU END

    shared_ptr<Tensor> result = make_shared<Tensor>(shape, requires_grad);
    shared_ptr<Tensor> sm_exp_broadcast = make_shared<Tensor>(shape, false);

    launch_softmax(shared_from_this(), sm_exp, sm_exp_broadcast, result, axis);

    if(requires_grad){
        result->parents.push_back(shared_from_this());
        result->backward_fn = [this, result, axis]() {
            auto dot = (result*result->grad)->sum(axis, true);
            this->grad += result*(result->grad-dot);
        };
    }

    return result;
}

// shared_ptr<Tensor> Tensor::log_softmax(int axis, bool keepdims) {
//     shared_ptr<Tensor> result = make_shared<Tensor>(shape, requires_grad);
//     for(int i=0; i<result->size(); i++){
//         // Convert linear index to multi-dimensional indices
//         int curr = i;
//         vector<int> indices(shape.size());
//         for(int x=shape.size()-1; x>=0; x--){
//             indices[x] = curr % shape[x];
//             curr /= shape[x];
//         }
        
//         // Find the maximum value along the axis for numerical stability
//         double max_val = at(i);
//         for(int j=0; j<shape[axis]; j++){
//             // Create indices for the current element along the axis
//             vector<int> max_indices = indices;
//             max_indices[axis] = j;
            
//             // Convert back to linear index
//             int max_idx = 0;
//             for(int x=0; x<shape.size(); x++){
//                 max_idx += max_indices[x] * strides[x];
//             }
            
//             max_val = std::max(max_val, (double)at(max_idx));
//         }
        
//         // Calculate the sum of exp(x - max_val) for all elements along the specified axis
//         double sum_exp = 0.0;
//         for(int j=0; j<shape[axis]; j++){
//             // Create indices for the current element along the axis
//             vector<int> sum_indices = indices;
//             sum_indices[axis] = j;
            
//             // Convert back to linear index
//             int sum_idx = 0;
//             for(int x=0; x<shape.size(); x++){
//                 sum_idx += sum_indices[x] * strides[x];
//             }
            
//             sum_exp += exp(at(sum_idx) - max_val);
//         }
        
//         // Set the result: log(exp(x - max_val) / sum(exp(x - max_val))) = (x - max_val) - log(sum_exp)
//         result->at(i) = at(i) - max_val - log(sum_exp);
//     }

//     if(requires_grad){
//         result->parents.push_back(shared_from_this());
//         result->backward_fn = [this, result, axis]() {
//             this->grad += result->grad - result->grad->sum(axis, true)->broadcast(shape, false);
//         };
//     }

//     return result;
// }

// shared_ptr<Tensor> Tensor::negative_log_likelihood(const shared_ptr<Tensor>& y_true) {
//     shared_ptr<Tensor> result = make_shared<Tensor>(shape, requires_grad);
//     for(int i=0; i<result->size(); i++){
//         result->at(i) = -log(at(i));
//     }
//     if(requires_grad){
//         result->parents.push_back(y_true);
//         result->parents.push_back(shared_from_this());
//         result->backward_fn = [y_true, result, this](){
//             if(y_true->requires_grad){
//                 y_true->grad += result->grad;
//             }
//             if(requires_grad){
//                 this->grad += result->grad;
//             }
//         };
//     }
//     return result;
// }

shared_ptr<Tensor> Tensor::cross_entropy(const shared_ptr<Tensor>& y_true, int axis, bool keepdims) {
    if(axis<0){
        axis+=shape.size();
    }

    if(axis>=shape.size()||axis<0){
        throw std::runtime_error("Invalid axis");
    }
    
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

    // CPU:
    // for(int i=0; i<result->size(); i++){
    //     int c=-1;
        
    //     // First pass: find max value along axis for numerical stability
    //     float max_val = -FLT_MAX;
    //     for(int j=0; j<shape[axis]; j++){
    //         int curr=i;
    //         int idx=0;
    //         for(int x=0; x<shape.size(); x++){
    //             if(x==axis){
    //                 idx+=j*strides[x];
    //             } else {
    //                 idx+=(curr/result->strides[x])*strides[x];
    //             }
    //             curr%=result->strides[x];
    //         }
    //         max_val = std::max(max_val, at(idx));
    //     }

    //     // Second pass: compute sum of exp(x - max) and find correct class
    //     for(int j=0; j<shape[axis]; j++){

    //         int curr=i;
    //         int idx=0;
    //         for(int x=0; x<shape.size(); x++){
    //             if(x==axis){
    //                 idx+=j*strides[x];
    //             } else {
    //                 idx+=(curr/result->strides[x])*strides[x];
    //             }
    //             curr%=result->strides[x];
    //         }

    //         result->at(i)+=exp(at(idx) - max_val);

    //         if(abs(y_true->at(idx)-1.0f)<=1e-5f){
    //             c=idx;
    //         }
    //     }
    //     if(c==-1){
    //         throw std::runtime_error("Invalid y_true. No '1' found in ground truth vector.");
    //     }
    //     // log(sum(exp(x - max))) + max - x[c] = log(sum(exp(x))) - x[c]
    //     result->at(i)=log(result->at(i)+1e-9f) + max_val - at(c);
    // }

    launch_cross_entropy(shared_from_this(), y_true, result, axis);

    if(requires_grad){
        result->parents.push_back(y_true);
        result->parents.push_back(shared_from_this());
        result->backward_fn = [y_true, result, axis, this](){
            this->grad+=shared_from_this()->softmax(axis)-y_true;

            // ASSUMES y_true DOESN'T REQUIRE GRADIENT
            if(y_true->requires_grad){
                throw std::runtime_error("y_true requires gradient. This is not supported.");
            }
        };
    }
    return result;
}

// Note: Makes a copy, but pytorch implementation doesn't, it only changes view. Shouldn't raise too big of issues tho.
shared_ptr<Tensor> Tensor::slice(int dim, int start, int end) {
    if(dim < 0) dim += shape.size();
    if(dim < 0 || dim >= shape.size()) {
        throw std::runtime_error("Invalid dimension for slice");
    }
    if(start < 0) start += shape[dim];
    if(end < 0) end += shape[dim];
    if(start < 0 || end > shape[dim] || start >= end) {
        throw std::runtime_error("Invalid slice indices");
    }
    
    vector<int> new_shape = shape;
    new_shape[dim] = end - start;
    
    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, requires_grad);
    
    for(int i = 0; i < result->size(); i++) {
        int curr = i;
        vector<int> indices(new_shape.size());
        for(int j = new_shape.size() - 1; j >= 0; j--) {
            indices[j] = curr % new_shape[j];
            curr /= new_shape[j];
        }
        indices[dim] += start;
        result->at(i) = at(indices);
    }
    
    if(requires_grad) {
        result->parents.push_back(shared_from_this());
        result->backward_fn = [this, result, dim, start]() {
            for(int i = 0; i < result->size(); i++) {
                int curr = i;
                vector<int> indices(result->shape.size());
                for(int j = result->shape.size() - 1; j >= 0; j--) {
                    indices[j] = curr % result->shape[j];
                    curr /= result->shape[j];
                }
                indices[dim] += start;
                int orig_idx = 0;
                for(int j = 0; j < indices.size(); j++) {
                    orig_idx += indices[j] * this->strides[j];
                }
                this->grad->at(orig_idx) += result->grad->at(i);
            }
        };
    }
    
    return result;
}

shared_ptr<Tensor> cat(const vector<shared_ptr<Tensor>>& tensors, int axis) {
    if(tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }
    
    if(axis < 0) axis += tensors[0]->shape.size();
    if(axis < 0 || axis >= tensors[0]->shape.size()) {
        throw std::runtime_error("Invalid axis for cat");
    }
    
    vector<int> base_shape = tensors[0]->shape;
    int total_dim = 0;
    bool any_requires_grad = false;
    
    for(const auto& t : tensors) {
        if(t->shape.size() != base_shape.size()) {
            throw std::runtime_error("All tensors must have same number of dimensions");
        }
        for(int i = 0; i < base_shape.size(); i++) {
            if(i != axis && t->shape[i] != base_shape[i]) {
                throw std::runtime_error("Tensor shapes must match except on concat axis");
            }
        }
        total_dim += t->shape[axis];
        if(t->requires_grad) any_requires_grad = true;
    }
    
    vector<int> new_shape = base_shape;
    new_shape[axis] = total_dim;
    
    shared_ptr<Tensor> result = make_shared<Tensor>(new_shape, any_requires_grad);
    
    int offset = 0;
    for(const auto& t : tensors) {
        for(int i = 0; i < t->size(); i++) {
            int curr = i;
            vector<int> indices(t->shape.size());
            for(int j = t->shape.size() - 1; j >= 0; j--) {
                indices[j] = curr % t->shape[j];
                curr /= t->shape[j];
            }
            indices[axis] += offset;
            result->at(indices) = t->at(i);
        }
        offset += t->shape[axis];
    }
    
    if(any_requires_grad) {
        for(const auto& t : tensors) {
            result->parents.push_back(t);
        }
        result->backward_fn = [tensors, result, axis]() {
            int offset = 0;
            for(const auto& t : tensors) {
                if(t->requires_grad) {
                    auto sliced_grad = result->grad->slice(axis, offset, offset + t->shape[axis]);
                    for(int i = 0; i < t->size(); i++) {
                        t->grad->at(i) += sliced_grad->at(i);
                    }
                }
                offset += t->shape[axis];
            }
        };
    }
    
    return result;
}

shared_ptr<Tensor> stack(const vector<shared_ptr<Tensor>>& tensors, int axis) {
    if(tensors.empty()) {
        throw std::runtime_error("Cannot stack empty tensor list");
    }
    
    vector<int> base_shape = tensors[0]->shape;
    int ndim = base_shape.size();
    
    if(axis < 0) axis += ndim + 1;
    if(axis < 0 || axis > ndim) {
        throw std::runtime_error("Invalid axis for stack");
    }
    
    for(const auto& t : tensors) {
        if(t->shape.size() != base_shape.size()) {
            throw std::runtime_error("All tensors must have same number of dimensions");
        }
        for(int i = 0; i < base_shape.size(); i++) {
            if(t->shape[i] != base_shape[i]) {
                throw std::runtime_error("All tensors must have same shape");
            }
        }
    }
    
    vector<shared_ptr<Tensor>> unsqueezed;
    for(const auto& t : tensors) {
        vector<int> new_shape;
        for(int i = 0; i < ndim + 1; i++) {
            if(i == axis) {
                new_shape.push_back(1);
            }
            if(i < axis) {
                new_shape.push_back(t->shape[i]);
            } else if(i > axis) {
                new_shape.push_back(t->shape[i - 1]);
            }
        }
        unsqueezed.push_back(t->reshape(new_shape));
    }
    
    return cat(unsqueezed, axis);
}

shared_ptr<Tensor> tril(const shared_ptr<Tensor>& A, float fill_value, int diagonal) {
    if(A->shape.size() < 2) {
        throw std::runtime_error("tril requires at least 2 dimensions");
    }
    
    shared_ptr<Tensor> result = make_shared<Tensor>(A->shape, A->requires_grad);
    
    int rows = A->shape[A->shape.size() - 2];
    int cols = A->shape[A->shape.size() - 1];
    
    for(int idx = 0; idx < result->size(); idx++) {
        int row = (idx / cols) % rows;
        int col = idx % cols;
        if(col <= row + diagonal) {
            result->at(idx) = A->at(idx);
        } else {
            result->at(idx) = fill_value;
        }
    }
    
    if(A->requires_grad) {
        result->parents.push_back(A);
        result->backward_fn = [A, result, rows, cols, diagonal]() {
            for(int idx = 0; idx < result->size(); idx++) {
                int row = (idx / cols) % rows;
                int col = idx % cols;
                if(col <= row + diagonal) {
                    A->grad->at(idx) += result->grad->at(idx);
                }
            }
        };
    }
    
    return result;
}

shared_ptr<Tensor> layer_norm(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& gamma, const shared_ptr<Tensor>& beta, float epsilon) {
    int axis = A->shape.size() - 1;
    int norm_size = A->shape[axis];
    
    if(gamma->size() != norm_size || beta->size() != norm_size) {
        throw std::runtime_error("gamma and beta must match last dimension size");
    }
    
    auto mean = A->mean(axis, true);
    auto var = A->variance_squared(axis, true);
    auto eps = make_shared<Tensor>(var->shape, epsilon, false);
    auto std_inv = (var + eps)->pow(-0.5f);
    auto x_norm = (A - mean->broadcast(A->shape)) * std_inv->broadcast(A->shape);
    auto result = x_norm * gamma->broadcast(A->shape) + beta->broadcast(A->shape);
    
    return result;
}

shared_ptr<Tensor> embedding(const shared_ptr<Tensor>& weight, const shared_ptr<Tensor>& indices) {
    if(weight->shape.size() != 2) {
        throw std::runtime_error("Embedding weight must be 2D (vocab_size x embedding_dim)");
    }
    
    int vocab_size = weight->shape[0];
    int embed_dim = weight->shape[1];
    
    for(int i = 0; i < indices->size(); i++) {
        int idx = (int)indices->at(i);
        if(idx < 0 || idx >= vocab_size) {
            throw std::runtime_error("Embedding index out of range");
        }
    }
    
    vector<int> result_shape = indices->shape;
    result_shape.push_back(embed_dim);
    shared_ptr<Tensor> result = make_shared<Tensor>(result_shape, weight->requires_grad);
    
    for(int idx = 0; idx < result->size(); idx++) {
        int i = idx / embed_dim;
        int j = idx % embed_dim;
        int row = (int)indices->at(i);
        result->at(idx) = weight->at(row * embed_dim + j);
    }
    
    if(weight->requires_grad) {
        result->parents.push_back(weight);
        result->backward_fn = [weight, result, indices, embed_dim]() {
            for(int idx = 0; idx < result->size(); idx++) {
                int i = idx / embed_dim;
                int j = idx % embed_dim;
                int row = (int)indices->at(i);
                weight->grad->at(row * embed_dim + j) += result->grad->at(idx);
            }
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