#include "tensor.h"
#include "tensor_kernels.h"
#include <cfloat>
#include <random>

// ==================== TensorImpl Implementation ====================

TensorImpl::TensorImpl() : data(nullptr), grad(nullptr), parents(), backward_fn(), device(DeviceType::CPU) {}

TensorImpl::TensorImpl(const vector<int>& shape, bool requires_grad, DeviceType device) 
    : shape(shape), requires_grad(requires_grad), device(device) {
    
    this->strides = compute_strides(shape);
    this->data = new float[size()]();

    if(requires_grad) {
        this->grad = make_shared<TensorImpl>(shape, false, device);
    } else {
        this->grad = nullptr;
    }

    this->parents = vector<shared_ptr<TensorImpl>>();
    this->backward_fn = [](){};
}

TensorImpl::TensorImpl(const vector<int>& shape, const vector<float>& data, bool requires_grad, DeviceType device) 
    : TensorImpl(shape, requires_grad, device) {
    
    if(data.size() != size()) {
        throw std::runtime_error("Data size mismatch");
    }

    for(int i = 0; i < size(); i++) {
        this->data[i] = data[i];
    }
}

TensorImpl::TensorImpl(const vector<int>& shape, float* data, bool requires_grad, DeviceType device) 
    : TensorImpl(shape, requires_grad, device) {
    for(int i = 0; i < size(); i++) {
        this->data[i] = data[i];
    }
}

TensorImpl::TensorImpl(const vector<int>& shape, float num, bool requires_grad, DeviceType device) 
    : TensorImpl(shape, requires_grad, device) {
    for(int i = 0; i < size(); i++) {
        this->data[i] = num;
    }
}

TensorImpl::TensorImpl(shared_ptr<TensorImpl> other) 
    : TensorImpl(other->shape, other->data, other->requires_grad, other->device) {
}

// Big 5 Implementation

TensorImpl::~TensorImpl() {
    delete[] data;
    data = nullptr;
}

TensorImpl::TensorImpl(const TensorImpl& other) 
    : requires_grad(other.requires_grad), device(other.device), shape(other.shape), strides(other.strides),
      parents(other.parents), backward_fn(other.backward_fn) {
    
    data = new float[size()]();
    for(int i = 0; i < size(); i++) {
        data[i] = other.data[i];
    }
    
    if(other.grad) {
        grad = make_shared<TensorImpl>(*other.grad);
    } else {
        grad = nullptr;
    }
}

TensorImpl& TensorImpl::operator=(const TensorImpl& other) {
    if(this != &other) {
        // Clean up existing data
        delete[] data;
        
        // Copy members
        requires_grad = other.requires_grad;
        device = other.device;
        shape = other.shape;
        strides = other.strides;
        parents = other.parents;
        backward_fn = other.backward_fn;
        
        // Deep copy data
        data = new float[size()]();
        for(int i = 0; i < size(); i++) {
            data[i] = other.data[i];
        }
        
        // Deep copy grad
        if(other.grad) {
            grad = make_shared<TensorImpl>(*other.grad);
        } else {
            grad = nullptr;
        }
    }
    return *this;
}

TensorImpl::TensorImpl(TensorImpl&& other) noexcept
    : requires_grad(other.requires_grad), device(other.device), shape(std::move(other.shape)), 
      strides(std::move(other.strides)), data(other.data), grad(std::move(other.grad)),
      parents(std::move(other.parents)), backward_fn(std::move(other.backward_fn)) {
    
    // Null out other's data pointer to prevent double deletion
    other.data = nullptr;
}

TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept {
    if(this != &other) {
        // Clean up existing data
        delete[] data;
        
        // Move members
        requires_grad = other.requires_grad;
        device = other.device;
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        data = other.data;
        grad = std::move(other.grad);
        parents = std::move(other.parents);
        backward_fn = std::move(other.backward_fn);
        
        // Null out other's data pointer
        other.data = nullptr;
    }
    return *this;
}

int TensorImpl::size() const {
    int cnt = 1;
    for(int i = 0; i < this->shape.size(); i++) {
        cnt *= this->shape[i];
    }
    return cnt;
}

vector<int> TensorImpl::compute_strides(const vector<int>& shape) {
    vector<int> strides(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

float& TensorImpl::at(vector<int> indices) {
    if(indices.size() < shape.size()) {
        throw std::runtime_error("Indices size mismatch");
    }

    if(indices.size() > shape.size()) {
        for(int i = 0; i < indices.size() - shape.size(); i++) {
            if(indices[i] != 0) {
                throw std::runtime_error("Shape mismatch");
            }
        }
    }
    
    int ind = 0;
    for(int i = indices.size() - shape.size(); i < indices.size(); i++) {
        if(indices[i] >= shape[i - indices.size() + shape.size()]) {
            throw std::runtime_error("Index out of bounds");
        }
        ind += strides[i - indices.size() + shape.size()] * indices[i];
    }
    
    return data[ind];
}

float& TensorImpl::at(int index) {
    if(index < 0 || index >= size()) {
        throw std::runtime_error("Index out of bounds");
    }
    return data[index];
}

void TensorImpl::print() {
    auto print_value = [](float val) {
        cout << val;
        if (val >= 0) cout << " ";
    };

    function<void(vector<int>&, int)> print_dim = [&](vector<int>& indices, int dim) {
        if (dim == shape.size()) {
            print_value(at(indices));
            return;
        }

        if (dim > 0) cout << "[";
        
        for (int i = 0; i < shape[dim]; i++) {
            indices[dim] = i;
            
            if (dim == shape.size() - 1) {
                print_value(at(indices));
                if (i < shape[dim] - 1) cout << ", ";
            } else {
                print_dim(indices, dim + 1);
                if (i < shape[dim] - 1) cout << ", ";
            }
        }
        
        if (dim > 0) cout << "]";
        
        if (dim == 0 && indices[0] < shape[0] - 1) cout << endl;
    };

    vector<int> indices(shape.size(), 0);
    
    cout << "Tensor(";
    for (int i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i < shape.size() - 1) cout << ", ";
    }
    cout << "):" << endl;
    
    print_dim(indices, 0);
    cout << endl;
}

// ==================== Tensor Implementation ====================

Tensor::Tensor() : impl(make_shared<TensorImpl>()) {}

Tensor::Tensor(shared_ptr<TensorImpl> impl) : impl(impl) {}

Tensor::Tensor(const vector<int>& shape, bool requires_grad, DeviceType device) 
    : impl(make_shared<TensorImpl>(shape, requires_grad, device)) {}

Tensor::Tensor(const vector<int>& shape, const vector<float>& data, bool requires_grad, DeviceType device) 
    : impl(make_shared<TensorImpl>(shape, data, requires_grad, device)) {}

Tensor::Tensor(const vector<int>& shape, float* data, bool requires_grad, DeviceType device) 
    : impl(make_shared<TensorImpl>(shape, data, requires_grad, device)) {}

Tensor::Tensor(const vector<int>& shape, float num, bool requires_grad, DeviceType device) 
    : impl(make_shared<TensorImpl>(shape, num, requires_grad, device)) {}

// Initializer list constructors
Tensor::Tensor(initializer_list<int> shape, bool requires_grad, DeviceType device)
    : Tensor(vector<int>(shape), requires_grad, device) {}

Tensor::Tensor(initializer_list<int> shape, initializer_list<float> data, bool requires_grad, DeviceType device)
    : Tensor(vector<int>(shape), vector<float>(data), requires_grad, device) {}

Tensor::Tensor(initializer_list<int> shape, float num, bool requires_grad, DeviceType device)
    : Tensor(vector<int>(shape), num, requires_grad, device) {}

// Big 5 Implementation

Tensor::Tensor(const Tensor& other) 
    : impl(other.impl) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if(this != &other) {
        impl = other.impl;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : impl(std::move(other.impl)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if(this != &other) {
        impl = std::move(other.impl);
    }
    return *this;
}

Tensor Tensor::clone() const {
    return Tensor(make_shared<TensorImpl>(*impl));
}

int Tensor::size() const {
    return impl->size();
}

float& Tensor::at(vector<int> indices) {
    return impl->at(indices);
}

float& Tensor::at(int index) {
    return impl->at(index);
}

float& Tensor::at(vector<int> indices) const {
    return impl->at(indices);
}

float& Tensor::at(int index) const {
    return impl->at(index);
}

vector<int> Tensor::compute_strides(const vector<int>& shape) {
    return impl->compute_strides(shape);
}

Tensor Tensor::grad() const {
    if (impl->grad) {
        return Tensor(impl->grad);
    }
    throw std::runtime_error("Gradient not available");
}

void Tensor::set_grad(const Tensor& g) {
    impl->grad = g.impl;
}

void Tensor::print() const {
    impl->print();
}

Tensor Tensor::reshape(const vector<int>& new_shape) const {
    int cnt = 1;
    for (int dim : new_shape) cnt *= dim;
    
    if(cnt != size()) {
        throw std::runtime_error("Reshape size mismatch");
    }

    Tensor result(new_shape, impl->data, impl->requires_grad, impl->device);

    if(impl->requires_grad) {
        shared_ptr<TensorImpl> this_impl = impl;
        shared_ptr<TensorImpl> result_impl = result.impl;
        
        result.impl->parents.push_back(this_impl);
        result.impl->backward_fn = [this_impl, result_impl]() {
            Tensor result_tensor(result_impl);
            Tensor reshaped_grad = result_tensor.grad().reshape(this_impl->shape);
            reshaped_grad.impl->requires_grad = false;
            
            Tensor this_tensor(this_impl);
            this_tensor.set_grad(this_tensor.grad() + reshaped_grad);
        };
    }

    return result;
}

void Tensor::backward() {
    // Topological sort using Kahn's algorithm
    unordered_map<shared_ptr<TensorImpl>, int> in_degree;
    queue<shared_ptr<TensorImpl>> q;
    unordered_set<shared_ptr<TensorImpl>> visited;
    
    function<void(shared_ptr<TensorImpl>)> count_in_degrees = [&](shared_ptr<TensorImpl> tensor) {
        if (visited.count(tensor)) return;
        visited.insert(tensor);
        
        in_degree[tensor] = 0;

        for (const auto& parent : tensor->parents) {
            count_in_degrees(parent);
            in_degree[parent]++;
        }
    };
    
    count_in_degrees(impl);
    
    for (const auto& pair : in_degree) {
        if (pair.second == 0) {
            q.push(pair.first);
        }
    }

    while (!q.empty()) {
        auto tensor = q.front();
        q.pop();
        
        if (tensor->backward_fn) {
            tensor->backward_fn();
        }
        
        for (const auto& parent : tensor->parents) {
            in_degree[parent]--;
            if (in_degree[parent] == 0) {
                q.push(parent);
            }
        }
    }
}

Tensor Tensor::sum(int axis, bool keepdims) const {
    vector<int> new_shape;
    for(int i = 0; i < impl->shape.size(); i++) {
        if(i == axis) {
            if(keepdims) {
                new_shape.push_back(1);
            }
        } else {
            new_shape.push_back(impl->shape[i]);
        }
    }

    Tensor result(new_shape, impl->requires_grad, impl->device);

    if(impl->device == DeviceType::CPU) {
        for(int i = 0; i < result.size(); i++) {
            for(int j = 0; j < impl->shape[axis]; j++) {
                int curr = i;
                int idx = 0;
                for(int x = 0; x < impl->shape.size(); x++) {
                    if(x == axis) {
                        idx += j * impl->strides[x];
                    } else {
                        idx += (curr / result.impl->strides[x]) * impl->strides[x];
                    }
                    curr %= result.impl->strides[x];
                }
                result.at(i) += impl->at(idx);
            }
        }
    } else {
        // For CUDA, we need to create shared_ptr wrappers for the launch function
        auto this_ptr = make_shared<Tensor>(*this);
        auto result_ptr = make_shared<Tensor>(result);
        launch_sum(this_ptr, result_ptr, axis);
    }

    if(impl->requires_grad) {
        shared_ptr<TensorImpl> this_impl = impl;
        shared_ptr<TensorImpl> result_impl = result.impl;
        
        result.impl->parents.push_back(this_impl);
        result.impl->backward_fn = [this_impl, result_impl]() {
            Tensor this_tensor(this_impl);
            Tensor result_tensor(result_impl);
            this_tensor.set_grad(this_tensor.grad() + result_tensor.grad().broadcast(this_impl->shape));
        };
    }

    return result;
}

bool is_broadcastable(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul) {
    if(matmul && (A_shape.size() < 2 || B_shape.size() < 2)) {
        return false;
    }

    if(matmul && A_shape[A_shape.size()-1] != B_shape[B_shape.size()-2]) {
        return false;
    }
    
    int d = max(A_shape.size(), B_shape.size());
    for(int i = 0; i < d - 2*matmul; i++) {
        int dim_A = (i < d - A_shape.size()) ? 1 : A_shape[i - (d - A_shape.size())];
        int dim_B = (i < d - B_shape.size()) ? 1 : B_shape[i - (d - B_shape.size())];
        
        if(dim_A == 1 || dim_B == 1) {
            continue;
        } else if(dim_A != dim_B) {
            return false;
        }
    }

    return true;
}

vector<int> get_broadcast_shape(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul) {
    if(!is_broadcastable(A_shape, B_shape, matmul)) {
        throw std::runtime_error("Broadcast shape mismatch");
    }

    vector<int> new_shape;
    int d = max(A_shape.size(), B_shape.size());

    for(int i = 0; i < d - 2*matmul; i++) {
        if(i < d - A_shape.size()) {
            new_shape.push_back(B_shape[i]);
        } else if(i < d - B_shape.size()) {
            new_shape.push_back(A_shape[i]);
        } else {
            if(A_shape[i - (d - A_shape.size())] == 1) {
                new_shape.push_back(B_shape[i - (d - B_shape.size())]);
            } else if(B_shape[i - (d - B_shape.size())] == 1) {
                new_shape.push_back(A_shape[i - (d - A_shape.size())]);
            } else if(A_shape[i - (d - A_shape.size())] == B_shape[i - (d - B_shape.size())]) {
                new_shape.push_back(A_shape[i - (d - A_shape.size())]);
            } else {
                throw invalid_argument("Shape mismatch");
            }
        }
    }

    return new_shape;
}

Tensor Tensor::broadcast(const vector<int>& new_shape, bool matmul) const {
    if(!is_broadcastable(impl->shape, new_shape, matmul)) {
        throw std::runtime_error("Broadcast shape mismatch");
    }

    bool same_shape = true;
    for(int i = 0; i < std::min(impl->shape.size(), new_shape.size()); i++) {
        if(impl->shape[i] != new_shape[i]) {
            same_shape = false;
            break;
        }
    }
    if(impl->shape.size() != new_shape.size()) same_shape = false;

    if(same_shape) {
        return Tensor(impl);  // Share impl, don't copy
    }

    Tensor result(new_shape, impl->requires_grad, impl->device);

    vector<int> padded_shape = impl->shape;
    vector<int> padded_strides = impl->strides;
    
    while(padded_shape.size() < new_shape.size()) {
        padded_shape.insert(padded_shape.begin(), 1);
        padded_strides.insert(padded_strides.begin(), 0);
    }

    if(impl->device == DeviceType::CPU) {
        for(int i = 0; i < result.size(); i++) {
            int curr = i;
            int idx = 0;
            for(int j = 0; j < new_shape.size() - 2*matmul; j++) {
                int dim = curr / result.impl->strides[j];
                curr %= result.impl->strides[j];
                
                if(padded_shape[j] == 1) {
                    idx += 0;
                } else {
                    idx += padded_strides[j] * dim;
                }
            }
            result.at(i) = impl->at(idx);
        }
    } else {
        auto this_ptr = make_shared<Tensor>(*this);
        auto result_ptr = make_shared<Tensor>(result);
        launch_broadcast(this_ptr, result_ptr, padded_shape, padded_strides, matmul);
    }

    if(impl->requires_grad) {
        shared_ptr<TensorImpl> this_impl = impl;
        shared_ptr<TensorImpl> result_impl = result.impl;
        
        result.impl->parents.push_back(this_impl);
        result.impl->backward_fn = [this_impl, result_impl]() {
            Tensor this_tensor(this_impl);
            Tensor result_tensor(result_impl);
            Tensor reduced_grad = result_tensor.grad().reduce_to_shape(this_impl->shape);
            reduced_grad.impl->requires_grad = false;
            this_tensor.set_grad(this_tensor.grad() + reduced_grad);
        };
    }

    return result;
}

Tensor Tensor::reduce_to_shape(const vector<int>& target_shape) const {
    if(target_shape.size() > impl->shape.size()) {
        throw std::runtime_error("Target shape size mismatch");
    }

    if(target_shape.size() == impl->shape.size()) {
        bool same = true;
        for(int i = 0; i < target_shape.size(); i++) {
            if(target_shape[i] != impl->shape[i]) {
                same = false;
                break;
            }
        }
        if(same) {
            return Tensor(impl);  // Share impl, don't copy
        }
    }
    
    vector<int> new_shape;
    if(target_shape.size() < impl->shape.size()) {
        for(int i = 0; i < impl->shape.size() - target_shape.size(); i++) {
            new_shape.push_back(1);
        }
    }

    for(int i = 0; i < target_shape.size(); i++) {
        new_shape.push_back(target_shape[i]);
    }

    Tensor result = *this;

    for(int i = 0; i < new_shape.size(); i++) {
        if(new_shape[i] == 1 && impl->shape[i] != 1) {
            result = result.sum(i, true);
        } else if(new_shape[i] == impl->shape[i]) {
            continue;
        } else {
            throw std::runtime_error("Target shape size mismatch");
        }
    }

    result = result.reshape(target_shape);
    return result;
}

Tensor operator+(const Tensor& A, const Tensor& B) {
    if(!is_broadcastable(A.shape(), B.shape(), false)) {
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A.shape(), B.shape(), false);

    Tensor new_A = A.broadcast(new_shape, false);
    Tensor new_B = B.broadcast(new_shape, false);

    if(new_A.size() != new_B.size()) {
        throw std::runtime_error("Broadcast size mismatch");
    }

    Tensor result(new_shape, A.requires_grad() || B.requires_grad(), A.device());

    if(A.device() == DeviceType::CPU) {
        for(int i = 0; i < new_A.size(); i++) {
            result.at(i) = new_A.at(i) + new_B.at(i);
        }
    } else {
        auto new_A_ptr = make_shared<Tensor>(new_A);
        auto new_B_ptr = make_shared<Tensor>(new_B);
        auto result_ptr = make_shared<Tensor>(result);
        launch_add(new_A_ptr, new_B_ptr, result_ptr);
    }

    if(new_A.requires_grad() || new_B.requires_grad()) {
        shared_ptr<TensorImpl> new_A_impl = new_A.impl;
        shared_ptr<TensorImpl> new_B_impl = new_B.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(new_A_impl);
        result.impl->parents.push_back(new_B_impl);

        result.impl->backward_fn = [new_A_impl, new_B_impl, result_impl]() {
            Tensor new_A(new_A_impl);
            Tensor new_B(new_B_impl);
            Tensor result(result_impl);

            if(new_A_impl->requires_grad && new_A_impl->grad != nullptr) {
                Tensor new_grad = new_A.grad() + result.grad().reduce_to_shape(new_A_impl->shape);
                new_A.set_grad(new_grad);
            }
            if(new_B_impl->requires_grad && new_B_impl->grad != nullptr) {
                Tensor new_grad = new_B.grad() + result.grad().reduce_to_shape(new_B_impl->shape);
                new_B.set_grad(new_grad);
            }
        };
    }

    return result;
}

Tensor operator-(const Tensor& A, const Tensor& B) {
    if(!is_broadcastable(A.shape(), B.shape(), false)) {
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A.shape(), B.shape(), false);

    Tensor new_A = A.broadcast(new_shape, false);
    Tensor new_B = B.broadcast(new_shape, false);

    Tensor result(new_shape, A.requires_grad() || B.requires_grad(), A.device());

    if(A.device() == DeviceType::CPU) {
        for(int i = 0; i < new_A.size(); i++) {
            result.at(i) = new_A.at(i) - new_B.at(i);
        }
    } else {
        auto new_A_ptr = make_shared<Tensor>(new_A);
        auto new_B_ptr = make_shared<Tensor>(new_B);
        auto result_ptr = make_shared<Tensor>(result);
        launch_subtract(new_A_ptr, new_B_ptr, result_ptr);
    }

    if(new_A.requires_grad() || new_B.requires_grad()) {
        shared_ptr<TensorImpl> new_A_impl = new_A.impl;
        shared_ptr<TensorImpl> new_B_impl = new_B.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(new_A_impl);
        result.impl->parents.push_back(new_B_impl);

        result.impl->backward_fn = [new_A_impl, new_B_impl, result_impl]() {
            Tensor new_A(new_A_impl);
            Tensor new_B(new_B_impl);
            Tensor result(result_impl);

            if(new_A_impl->requires_grad && new_A_impl->grad != nullptr) {
                Tensor new_grad = new_A.grad() + result.grad().reduce_to_shape(new_A_impl->shape);
                new_A.set_grad(new_grad);
            }
            if(new_B_impl->requires_grad && new_B_impl->grad != nullptr) {
                Tensor new_grad = new_B.grad() - result.grad().reduce_to_shape(new_B_impl->shape);
                new_B.set_grad(new_grad);
            }
        };
    }
    
    return result;
}

Tensor operator*(const Tensor& A, const Tensor& B) {
    if(!is_broadcastable(A.shape(), B.shape(), false)) {
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A.shape(), B.shape(), false);

    Tensor new_A = A.broadcast(new_shape, false);
    Tensor new_B = B.broadcast(new_shape, false);
    
    Tensor result(new_shape, A.requires_grad() || B.requires_grad(), A.device());

    if(A.device() == DeviceType::CPU) {
        for(int i = 0; i < new_A.size(); i++) {
            result.at(i) = new_A.at(i) * new_B.at(i);
        }
    } else {
        auto new_A_ptr = make_shared<Tensor>(new_A);
        auto new_B_ptr = make_shared<Tensor>(new_B);
        auto result_ptr = make_shared<Tensor>(result);
        launch_multiply(new_A_ptr, new_B_ptr, result_ptr);
    }

    if(new_A.requires_grad() || new_B.requires_grad()) {
        shared_ptr<TensorImpl> new_A_impl = new_A.impl;
        shared_ptr<TensorImpl> new_B_impl = new_B.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(new_A_impl);
        result.impl->parents.push_back(new_B_impl);

        result.impl->backward_fn = [new_A_impl, new_B_impl, result_impl]() {
            Tensor new_A(new_A_impl);
            Tensor new_B(new_B_impl);
            Tensor result(result_impl);

            if(new_A_impl->requires_grad && new_A_impl->grad != nullptr) {
                Tensor new_grad = new_A.grad() + result.grad() * new_B;
                new_A.set_grad(new_grad);
            }
            if(new_B_impl->requires_grad && new_B_impl->grad != nullptr) {
                Tensor new_grad = new_B.grad() + result.grad() * new_A;
                new_B.set_grad(new_grad);
            }
        };
    }

    return result;
}

Tensor operator/(const Tensor& A, const Tensor& B) {
    if(!is_broadcastable(A.shape(), B.shape(), false)) {
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A.shape(), B.shape(), false);

    Tensor new_A = A.broadcast(new_shape, false);
    Tensor new_B = B.broadcast(new_shape, false);
    
    Tensor result(new_shape, A.requires_grad() || B.requires_grad(), A.device());

    if(A.device() == DeviceType::CPU) {
        for(int i = 0; i < new_A.size(); i++) {
            result.at(i) = new_A.at(i) / new_B.at(i);
        }
    } else {
        auto new_A_ptr = make_shared<Tensor>(new_A);
        auto new_B_ptr = make_shared<Tensor>(new_B);
        auto result_ptr = make_shared<Tensor>(result);
        launch_divide(new_A_ptr, new_B_ptr, result_ptr);
    }

    if(new_A.requires_grad() || new_B.requires_grad()) {
        shared_ptr<TensorImpl> new_A_impl = new_A.impl;
        shared_ptr<TensorImpl> new_B_impl = new_B.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(new_A_impl);
        result.impl->parents.push_back(new_B_impl);
        
        result.impl->backward_fn = [new_A_impl, new_B_impl, result_impl]() {
            Tensor new_A(new_A_impl);
            Tensor new_B(new_B_impl);
            Tensor result(result_impl);

            if(new_A_impl->requires_grad) {
                Tensor new_grad = new_A.grad() + result.grad() / new_B;
                new_A.set_grad(new_grad);
            }
            if(new_B_impl->requires_grad) {
                Tensor new_grad = new_B.grad() + result.grad() * (-new_A / (new_B * new_B));
                new_B.set_grad(new_grad);
            }
        };
    }

    return result;
}
    
Tensor& operator+=(Tensor& A, const Tensor& B) {
    A = A + B;
    return A;
}

Tensor& operator-=(Tensor& A, const Tensor& B) {
    A = A - B;
    return A;
}

Tensor& operator*=(Tensor& A, const Tensor& B) {
    A = A * B;
    return A;
}

Tensor& operator/=(Tensor& A, const Tensor& B) {
    A = A / B;
    return A;
}

Tensor operator+(const Tensor& A, float B) {
    return A + Tensor(vector<int>{1}, vector<float>{B}, true, A.device());
}

Tensor operator+(float A, const Tensor& B) {
    return Tensor(vector<int>{1}, vector<float>{A}, true, B.device()) + B;
}

Tensor operator-(const Tensor& A, float B) {
    return A - Tensor(vector<int>{1}, vector<float>{B}, true, A.device());
}

Tensor operator-(float A, const Tensor& B) {
    return Tensor(vector<int>{1}, vector<float>{A}, true, B.device()) - B;
}

Tensor operator*(const Tensor& A, float B) {
    return A * Tensor(vector<int>{1}, vector<float>{B}, true, A.device());
}

Tensor operator*(float A, const Tensor& B) {
    return Tensor(vector<int>{1}, vector<float>{A}, true, B.device()) * B;
}

Tensor operator/(const Tensor& A, float B) {
    return A / Tensor(vector<int>{1}, vector<float>{B}, true, A.device());
}

Tensor operator/(float A, const Tensor& B) {
    return Tensor(vector<int>{1}, vector<float>{A}, true, B.device()) / B;
}

Tensor operator-(const Tensor& A) {
    return -1.0f * A;
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    if(A.shape().size() < 2 || B.shape().size() < 2) {
        throw invalid_argument("Tensor must have at least 2 dimensions");
    }

    if(A.shape()[A.shape().size()-1] != B.shape()[B.shape().size()-2]) {
        throw invalid_argument("Shape mismatch");
    }

    if(!is_broadcastable(A.shape(), B.shape(), true)) {
        throw invalid_argument("Shape mismatch");
    }

    vector<int> new_shape = get_broadcast_shape(A.shape(), B.shape(), true);

    vector<int> new_shape_A(new_shape);
    new_shape_A.push_back(A.shape()[A.shape().size()-2]);
    new_shape_A.push_back(A.shape()[A.shape().size()-1]);
    vector<int> new_shape_B(new_shape);
    new_shape_B.push_back(B.shape()[B.shape().size()-2]);
    new_shape_B.push_back(B.shape()[B.shape().size()-1]);

    Tensor new_A = A.broadcast(new_shape_A);
    Tensor new_B = B.broadcast(new_shape_B);
    
    vector<int> matmul_output_shape;
    for(int i = 0; i < new_shape.size(); i++) {
        matmul_output_shape.push_back(new_shape[i]);
    }
    matmul_output_shape.push_back(new_shape_A[new_shape_A.size()-2]);
    matmul_output_shape.push_back(new_shape_B[new_shape_B.size()-1]);

    Tensor result(matmul_output_shape, A.requires_grad() || B.requires_grad(), A.device());
    
    if(A.device() == DeviceType::CPU) {
        for(int i = 0; i < result.size(); i++) {
            int ind_A = 0;
            int ind_B = 0;
            int curr_A = i;
            int curr_B = i;

            for(int j = 0; j < new_shape_A.size() - 2; j++) {
                ind_A += (curr_A / result.impl->strides[j]) * new_A.impl->strides[j];
                curr_A %= result.impl->strides[j];
            }

            ind_A += (curr_A / result.impl->strides[result.impl->strides.size()-2]) * new_A.impl->strides[new_A.impl->strides.size()-2];
            curr_A %= result.impl->strides[result.impl->strides.size()-1];
            
            for(int j = 0; j < new_shape_B.size() - 2; j++) {
                ind_B += (curr_B / result.impl->strides[j]) * new_B.impl->strides[j];
                curr_B %= result.impl->strides[j];
            }

            curr_B %= result.impl->strides[result.impl->strides.size()-2];
            ind_B += (curr_B / result.impl->strides[result.impl->strides.size()-1]) * new_B.impl->strides[new_B.impl->strides.size()-1];
            
            for(int j = 0; j < new_shape_A[new_shape_A.size()-1]; j++) {
                result.at(i) += new_A.at(ind_A) * new_B.at(ind_B);
                ind_A += new_A.impl->strides[new_A.impl->strides.size()-1];
                ind_B += new_B.impl->strides[new_B.impl->strides.size()-2];
            }
        }
    } else {
        auto new_A_ptr = make_shared<Tensor>(new_A);
        auto new_B_ptr = make_shared<Tensor>(new_B);
        auto result_ptr = make_shared<Tensor>(result);
        launch_matmul(new_A_ptr, new_B_ptr, result_ptr);
    }
    
    if(new_A.requires_grad() || new_B.requires_grad()) {
        shared_ptr<TensorImpl> new_A_impl = new_A.impl;
        shared_ptr<TensorImpl> new_B_impl = new_B.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(new_A_impl);
        result.impl->parents.push_back(new_B_impl);
        
        result.impl->backward_fn = [new_A_impl, new_B_impl, result_impl]() {
            Tensor new_A(new_A_impl);
            Tensor new_B(new_B_impl);
            Tensor result(result_impl);

            if(new_A_impl->requires_grad) {
                auto B_transposed = new_B.transpose(new_B_impl->shape.size()-2, new_B_impl->shape.size()-1);
                Tensor new_grad = new_A.grad() + matmul(result.grad(), B_transposed);
                new_A.set_grad(new_grad);
            }
            if(new_B_impl->requires_grad) {
                auto A_transposed = new_A.transpose(new_A_impl->shape.size()-2, new_A_impl->shape.size()-1);
                Tensor new_grad = new_B.grad() + matmul(A_transposed, result.grad());
                new_B.set_grad(new_grad);
            }
        };
    }

    return result;
}

Tensor Tensor::transpose(int dim1, int dim2) const {
    if(impl->shape.size() < 2) {
        throw std::runtime_error("Tensor must have at least 2 dimensions");
    }

    int actual_dim1 = dim1;
    int actual_dim2 = dim2;

    if((actual_dim1 > 0 && actual_dim1 >= impl->shape.size()) ||
       (actual_dim1 < 0 && actual_dim1 < -(int)impl->shape.size()) ||
       (actual_dim2 > 0 && actual_dim2 >= impl->shape.size()) ||
       (actual_dim2 < 0 && actual_dim2 < -(int)impl->shape.size())) {
        throw std::runtime_error("Invalid dimension");
    }

    if(actual_dim1 == actual_dim2) {
        return Tensor(impl);  // Share impl, don't copy
    }

    if(actual_dim1 < 0) actual_dim1 += impl->shape.size();
    if(actual_dim2 < 0) actual_dim2 += impl->shape.size();

    vector<int> new_shape(impl->shape);
    swap(new_shape[actual_dim1], new_shape[actual_dim2]);

    Tensor result(new_shape, impl->data, impl->requires_grad, impl->device);

    if(impl->device == DeviceType::CPU) {
        for(int i = 0; i < result.size(); i++) {
            int ind = 0;
            int curr = i;
            int mag_dim1;
            int mag_dim2;
            for(int j = 0; j < new_shape.size(); j++) {
                if(j == actual_dim1) {
                    mag_dim1 = curr / result.impl->strides[j];
                } else if(j == actual_dim2) {
                    mag_dim2 = curr / result.impl->strides[j];
                }
                ind += (curr / result.impl->strides[j]) * impl->strides[j];
                curr %= result.impl->strides[j];
            }
            ind -= mag_dim1 * impl->strides[actual_dim1];
            ind += mag_dim2 * impl->strides[actual_dim1];
            ind += mag_dim1 * impl->strides[actual_dim2];
            ind -= mag_dim2 * impl->strides[actual_dim2];
            result.at(i) = impl->at(ind);
        }
    } else {
        auto this_ptr = make_shared<Tensor>(*this);
        auto result_ptr = make_shared<Tensor>(result);
        launch_transpose(this_ptr, result_ptr, actual_dim1, actual_dim2);
    }

    if(impl->requires_grad) {
        shared_ptr<TensorImpl> this_impl = impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(this_impl);

        result.impl->backward_fn = [actual_dim1, actual_dim2, this_impl, result_impl]() {
            Tensor this_tensor(this_impl);
            Tensor result_tensor(result_impl);
            this_tensor.set_grad(this_tensor.grad() + result_tensor.grad().transpose(actual_dim1, actual_dim2));
        };
    }

    return result;
}

Tensor Tensor::pow(float exponent) const {
    Tensor result(impl->shape, impl->requires_grad, impl->device);
    
    if(impl->device == DeviceType::CPU) {
        for(int i = 0; i < result.size(); i++) {
            result.at(i) = std::pow(impl->at(i), exponent);
        }
    } else {
        auto this_ptr = make_shared<Tensor>(*this);
        auto result_ptr = make_shared<Tensor>(result);
        launch_pow(this_ptr, result_ptr, exponent);
    }

    if(impl->requires_grad) {
        shared_ptr<TensorImpl> this_impl = impl;
        shared_ptr<TensorImpl> result_impl = result.impl;
        
        result.impl->parents.push_back(this_impl);
        result.impl->backward_fn = [exponent, this_impl, result_impl]() {
            Tensor this_tensor(this_impl);
            Tensor result_tensor(result_impl);
            this_tensor.set_grad(this_tensor.grad() + result_tensor.grad() * exponent * this_tensor.pow(exponent - 1.0f));
        };
    }
    return result;
}

Tensor Tensor::mean(int axis, bool keepdims) const {
    int N = impl->shape[axis];
    return sum(axis, keepdims) / N;
}

Tensor Tensor::variance_squared(int axis, bool keepdims) const {
    Tensor centered = *this - mean(axis, keepdims).broadcast(impl->shape, false);
    Tensor centered_squared = centered * centered;
    return centered_squared.mean(axis, keepdims);
}

Tensor relu(const Tensor& A) {
    Tensor result(A.shape(), A.requires_grad(), A.device());
    
    if(A.device() == DeviceType::CPU) {
        for(int i = 0; i < result.size(); i++) {
            result.at(i) = std::max(0.0f, A.at(i));
        }
    } else {
        auto A_ptr = make_shared<Tensor>(A);
        auto result_ptr = make_shared<Tensor>(result);
        launch_relu(A_ptr, result_ptr);
    }

    if(A.requires_grad()) {
        shared_ptr<TensorImpl> A_impl = A.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(A_impl);
        result.impl->backward_fn = [A_impl, result_impl]() {
            Tensor A(A_impl);
            Tensor result(result_impl);
            
            Tensor grad_mask(result_impl->shape, false, A_impl->device);
            for(int i = 0; i < result.size(); i++) {
                grad_mask.at(i) = result.at(i) > 0.0f ? 1.0f : 0.0f;
            }
            A.set_grad(A.grad() + result.grad() * grad_mask);
        };
    }
    return result;
}

Tensor sigmoid(const Tensor& A) {
    Tensor result(A.shape(), A.requires_grad(), A.device());
    
    if(A.device() == DeviceType::CPU) {
        for(int i = 0; i < result.size(); i++) {
            result.at(i) = 1.0f / (1.0f + exp(-A.at(i)));
        }
    } else {
        auto A_ptr = make_shared<Tensor>(A);
        auto result_ptr = make_shared<Tensor>(result);
        launch_sigmoid(A_ptr, result_ptr);
    }

    if(A.requires_grad()) {
        shared_ptr<TensorImpl> A_impl = A.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(A_impl);
        result.impl->backward_fn = [A_impl, result_impl]() {
            Tensor A(A_impl);
            Tensor result(result_impl);
            A.set_grad(A.grad() + result.grad() * result * (1.0f - result));
        };
    }
    return result;
}

Tensor tanh_op(const Tensor& A) {
    Tensor result(A.shape(), A.requires_grad(), A.device());
    
    if(A.device() == DeviceType::CPU) {
        for(int i = 0; i < result.size(); i++) {
            result.at(i) = (exp(A.at(i)) - exp(-A.at(i))) / (exp(A.at(i)) + exp(-A.at(i)));
        }
    } else {
        auto A_ptr = make_shared<Tensor>(A);
        auto result_ptr = make_shared<Tensor>(result);
        launch_tanh(A_ptr, result_ptr);
    }

    if(A.requires_grad()) {
        shared_ptr<TensorImpl> A_impl = A.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(A_impl);
        result.impl->backward_fn = [A_impl, result_impl]() {
            Tensor A(A_impl);
            Tensor result(result_impl);
            A.set_grad(A.grad() + result.grad() * (1.0f - result * result));
        };
    }
    return result;
}

Tensor dropout(const Tensor& A, float p, bool training) {
    Tensor mask(A.shape(), false, A.device());
    
    if(!training || p == 0.0f) {
        for(int i = 0; i < mask.size(); i++) {
            mask.at(i) = 1.0f;
        }
    } else {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - p);
        float scale = 1.0f / (1.0f - p);
        
        for(int i = 0; i < mask.size(); i++) {
            mask.at(i) = dist(gen) ? scale : 0.0f;
        }
    }

    Tensor result = A * mask;
    
    if(A.requires_grad()) {
        shared_ptr<TensorImpl> A_impl = A.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;
        shared_ptr<TensorImpl> mask_impl = mask.impl;

        result.impl->parents.push_back(A_impl);
        result.impl->backward_fn = [A_impl, result_impl, mask_impl]() {
            Tensor A(A_impl);
            Tensor result(result_impl);
            Tensor mask(mask_impl);
            A.set_grad(A.grad() + result.grad() * mask);
        };
    }
    return result;
}

Tensor softmax(const Tensor& A, int axis) {
    int actual_axis = axis;
    if(actual_axis < 0) {
        actual_axis += A.shape().size();
    }
    if(actual_axis >= (int)A.shape().size() || actual_axis < 0) {
        throw std::runtime_error("Invalid axis");
    }

    vector<int> sm_exp_shape;
    for(int i = 0; i < A.shape().size(); i++) {
        if(i == actual_axis) {
            sm_exp_shape.push_back(1);
        } else {
            sm_exp_shape.push_back(A.shape()[i]);
        }
    }

    Tensor sm_exp(sm_exp_shape, A.requires_grad(), A.device());
    Tensor result(A.shape(), A.requires_grad(), A.device());
    Tensor sm_exp_broadcast(A.shape(), false, A.device());

    if(A.device() == DeviceType::CPU) {
        float mx = -FLT_MAX;
        for(int i = 0; i < A.size(); i++) {
            mx = std::max(mx, A.at(i));
        }

        for(int i = 0; i < sm_exp.size(); i++) {
            for(int j = 0; j < A.shape()[actual_axis]; j++) {
                int curr = i;
                int idx = 0;
                for(int x = 0; x < A.shape().size(); x++) {
                    if(x == actual_axis) {
                        idx += j * A.impl->strides[x];
                    } else {
                        idx += (curr / sm_exp.impl->strides[x]) * A.impl->strides[x];
                    }
                    curr %= sm_exp.impl->strides[x];
                }
                sm_exp.at(i) += exp(A.at(idx) - mx);
            }
        }

        sm_exp_broadcast = sm_exp.broadcast(A.shape(), false);

        for(int i = 0; i < result.size(); i++) {
            result.at(i) = exp(A.at(i) - mx) / sm_exp_broadcast.at(i);
        }
    } else {
        auto A_ptr = make_shared<Tensor>(A);
        auto sm_exp_ptr = make_shared<Tensor>(sm_exp);
        auto sm_exp_broadcast_ptr = make_shared<Tensor>(sm_exp_broadcast);
        auto result_ptr = make_shared<Tensor>(result);
        launch_softmax(A_ptr, sm_exp_ptr, sm_exp_broadcast_ptr, result_ptr, actual_axis);
    }

    if(A.requires_grad()) {
        shared_ptr<TensorImpl> A_impl = A.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(A_impl);
        result.impl->backward_fn = [A_impl, result_impl, actual_axis]() {
            Tensor A(A_impl);
            Tensor result(result_impl);
            Tensor dot = (result * result.grad()).sum(actual_axis, true);
            A.set_grad(A.grad() + result * (result.grad() - dot));
        };
    }

    return result;
}

Tensor Tensor::cross_entropy(const Tensor& y_true, int axis, bool keepdims) const {
    int actual_axis = axis;
    if(actual_axis < 0) {
        actual_axis += impl->shape.size();
    }

    if(actual_axis >= impl->shape.size() || actual_axis < 0) {
        throw std::runtime_error("Invalid axis");
    }
    
    vector<int> new_shape;
    for(int i = 0; i < impl->shape.size(); i++) {
        if(i == actual_axis) {
            if(keepdims) {
                new_shape.push_back(1);
            }
        } else {
            new_shape.push_back(impl->shape[i]);
        }
    }

    Tensor result(new_shape, impl->requires_grad, impl->device);

    if(impl->device == DeviceType::CPU) {
        for(int i = 0; i < result.size(); i++) {
            int c = -1;
            
            float max_val = -FLT_MAX;
            for(int j = 0; j < impl->shape[actual_axis]; j++) {
                int curr = i;
                int idx = 0;
                for(int x = 0; x < impl->shape.size(); x++) {
                    if(x == actual_axis) {
                        idx += j * impl->strides[x];
                    } else {
                        idx += (curr / result.impl->strides[x]) * impl->strides[x];
                    }
                    curr %= result.impl->strides[x];
                }
                max_val = std::max(max_val, impl->at(idx));
            }

            for(int j = 0; j < impl->shape[actual_axis]; j++) {
                int curr = i;
                int idx = 0;
                for(int x = 0; x < impl->shape.size(); x++) {
                    if(x == actual_axis) {
                        idx += j * impl->strides[x];
                    } else {
                        idx += (curr / result.impl->strides[x]) * impl->strides[x];
                    }
                    curr %= result.impl->strides[x];
                }

                result.at(i) += exp(impl->at(idx) - max_val);

                if(abs(y_true.at(idx) - 1.0f) <= 1e-5f) {
                    c = idx;
                }
            }
            if(c == -1) {
                throw std::runtime_error("Invalid y_true. No '1' found in ground truth vector.");
            }
            result.at(i) = log(result.at(i) + 1e-9f) + max_val - impl->at(c);
        }
    } else {
        auto this_ptr = make_shared<Tensor>(*this);
        auto y_true_ptr = make_shared<Tensor>(y_true);
        auto result_ptr = make_shared<Tensor>(result);
        launch_cross_entropy(this_ptr, y_true_ptr, result_ptr, actual_axis);
    }

    if(impl->requires_grad) {
        shared_ptr<TensorImpl> this_impl = impl;
        shared_ptr<TensorImpl> y_true_impl = y_true.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(y_true_impl);
        result.impl->parents.push_back(this_impl);
        result.impl->backward_fn = [y_true_impl, result_impl, actual_axis, this_impl]() {
            Tensor this_tensor(this_impl);
            Tensor y_true(y_true_impl);
            this_tensor.set_grad(this_tensor.grad() + softmax(this_tensor, actual_axis) - y_true);

            if(y_true_impl->requires_grad) {
                throw std::runtime_error("y_true requires gradient. This is not supported.");
            }
        };
    }
    return result;
}

// TODO: Implement slice + Storage so slice makes new TensorImpl but same Storage just diff view
Tensor Tensor::slice(int dim, int start, int end) const {
    int actual_dim = dim;
    if(actual_dim < 0) actual_dim += impl->shape.size();
    if(actual_dim < 0 || actual_dim >= impl->shape.size()) {
        throw std::runtime_error("Invalid dimension for slice");
    }
    int actual_start = start;
    int actual_end = end;
    if(actual_start < 0) actual_start += impl->shape[actual_dim];
    if(actual_end < 0) actual_end += impl->shape[actual_dim];
    if(actual_start < 0 || actual_end > impl->shape[actual_dim] || actual_start >= actual_end) {
        throw std::runtime_error("Invalid slice indices");
    }
    
    vector<int> new_shape = impl->shape;
    new_shape[actual_dim] = actual_end - actual_start;
    
    Tensor result(new_shape, impl->requires_grad, impl->device);
    
    for(int i = 0; i < result.size(); i++) {
        int curr = i;
        vector<int> indices(new_shape.size());
        for(int j = new_shape.size() - 1; j >= 0; j--) {
            indices[j] = curr % new_shape[j];
            curr /= new_shape[j];
        }
        indices[actual_dim] += actual_start;
        result.at(i) = impl->at(indices);
    }
    
    if(impl->requires_grad) {
        shared_ptr<TensorImpl> this_impl = impl;
        shared_ptr<TensorImpl> result_impl = result.impl;

        result.impl->parents.push_back(this_impl);
        result.impl->backward_fn = [this_impl, result_impl, actual_dim, actual_start]() {
            Tensor this_tensor(this_impl);
            Tensor result(result_impl);
            
            for(int i = 0; i < result.size(); i++) {
                int curr = i;
                vector<int> indices(result_impl->shape.size());
                for(int j = result_impl->shape.size() - 1; j >= 0; j--) {
                    indices[j] = curr % result_impl->shape[j];
                    curr /= result_impl->shape[j];
                }
                indices[actual_dim] += actual_start;
                int orig_idx = 0;
                for(int j = 0; j < indices.size(); j++) {
                    orig_idx += indices[j] * this_impl->strides[j];
                }
                this_tensor.grad().at(orig_idx) += result.grad().at(i);
            }
        };
    }
    
    return result;
}

Tensor cat(const vector<Tensor>& tensors, int axis) {
    if(tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }
    
    int actual_axis = axis;
    if(actual_axis < 0) actual_axis += tensors[0].shape().size();
    if(actual_axis < 0 || actual_axis >= tensors[0].shape().size()) {
        throw std::runtime_error("Invalid axis for cat");
    }
    
    vector<int> base_shape = tensors[0].shape();
    DeviceType device = tensors[0].device();
    int total_dim = 0;
    bool any_requires_grad = false;
    
    for(const auto& t : tensors) {
        if(t.shape().size() != base_shape.size()) {
            throw std::runtime_error("All tensors must have same number of dimensions");
        }
        for(int i = 0; i < base_shape.size(); i++) {
            if(i != actual_axis && t.shape()[i] != base_shape[i]) {
                throw std::runtime_error("Tensor shapes must match except on concat axis");
            }
        }
        total_dim += t.shape()[actual_axis];
        if(t.requires_grad()) any_requires_grad = true;
    }
    
    vector<int> new_shape = base_shape;
    new_shape[actual_axis] = total_dim;
    
    Tensor result(new_shape, any_requires_grad, device);
    
    int offset = 0;
    for(const auto& t : tensors) {
        for(int i = 0; i < t.size(); i++) {
            int curr = i;
            vector<int> indices(t.shape().size());
            for(int j = t.shape().size() - 1; j >= 0; j--) {
                indices[j] = curr % t.shape()[j];
                curr /= t.shape()[j];
            }
            indices[actual_axis] += offset;
            result.at(indices) = t.at(i);
        }
        offset += t.shape()[actual_axis];
    }
    
    if(any_requires_grad) {
        vector<shared_ptr<TensorImpl>> tensor_impls;
        for(const auto& t : tensors) {
            result.impl->parents.push_back(t.impl);
            tensor_impls.push_back(t.impl);
        }
        
        shared_ptr<TensorImpl> result_impl = result.impl;
        
        result.impl->backward_fn = [tensor_impls, result_impl, actual_axis]() {
            Tensor result(result_impl);
            int offset = 0;
            for(const auto& t_impl : tensor_impls) {
                if(t_impl->requires_grad) {
                    Tensor t(t_impl);
                    Tensor sliced_grad = result.grad().slice(actual_axis, offset, offset + t_impl->shape[actual_axis]);
                    for(int i = 0; i < t.size(); i++) {
                        t.grad().at(i) += sliced_grad.at(i);
                    }
                }
                offset += t_impl->shape[actual_axis];
            }
        };
    }
    
    return result;
}

Tensor stack(const vector<Tensor>& tensors, int axis) {
    if(tensors.empty()) {
        throw std::runtime_error("Cannot stack empty tensor list");
    }
    
    vector<int> base_shape = tensors[0].shape();
    int ndim = base_shape.size();
    
    int actual_axis = axis;
    if(actual_axis < 0) actual_axis += ndim + 1;
    if(actual_axis < 0 || actual_axis > ndim) {
        throw std::runtime_error("Invalid axis for stack");
    }
    
    for(const auto& t : tensors) {
        if(t.shape().size() != base_shape.size()) {
            throw std::runtime_error("All tensors must have same number of dimensions");
        }
        for(int i = 0; i < base_shape.size(); i++) {
            if(t.shape()[i] != base_shape[i]) {
                throw std::runtime_error("All tensors must have same shape");
            }
        }
    }
    
    vector<Tensor> unsqueezed;
    for(const auto& t : tensors) {
        vector<int> new_shape;
        for(int i = 0; i < ndim + 1; i++) {
            if(i == actual_axis) {
                new_shape.push_back(1);
            }
            if(i < actual_axis) {
                new_shape.push_back(t.shape()[i]);
            } else if(i > actual_axis) {
                new_shape.push_back(t.shape()[i - 1]);
            }
        }
        unsqueezed.push_back(t.reshape(new_shape));
    }
    
    return cat(unsqueezed, actual_axis);
}

Tensor Tensor::masked_fill(const Tensor& mask, float value) const {
    if(mask.size() != size()) {
        throw std::runtime_error("Mask size must match tensor size");
    }
    
    Tensor result(impl->shape, impl->requires_grad, impl->device);
    
    for(int idx = 0; idx < result.size(); idx++) {
        if(mask.at(idx) != 0.0f) {
            result.at(idx) = value;
        } else {
            result.at(idx) = impl->at(idx);
        }
    }
    
    if(impl->requires_grad) {
        shared_ptr<TensorImpl> this_impl = impl;
        shared_ptr<TensorImpl> result_impl = result.impl;
        shared_ptr<TensorImpl> mask_impl = mask.impl;

        result.impl->parents.push_back(this_impl);
        result.impl->backward_fn = [this_impl, result_impl, mask_impl]() {
            Tensor this_tensor(this_impl);
            Tensor result(result_impl);
            Tensor mask(mask_impl);
            
            for(int idx = 0; idx < result.size(); idx++) {
                if(mask.at(idx) == 0.0f) {
                    this_tensor.grad().at(idx) += result.grad().at(idx);
                }
            }
        };
    }
    
    return result;
}

Tensor tril(int rows, int cols, DeviceType device) {
    Tensor result(vector<int>{rows, cols}, 0.0f, false, device);
    
    for(int idx = 0; idx < result.size(); idx++) {
        int row = idx / cols;
        int col = idx % cols;
        if(col <= row) {
            result.at(idx) = 1.0f;
        }
    }
    
    return result;
}

Tensor arange(float start, float end, float step, DeviceType device) {
    if(step == 0.0f) {
        throw std::runtime_error("arange step cannot be zero");
    }
    int n = (int)ceil((end - start) / step);
    if(n <= 0) n = 0;
    
    Tensor result(vector<int>{n}, 0.0f, false, device);
    for(int i = 0; i < n; i++) {
        result.at(i) = start + i * step;
    }
    return result;
}

Tensor multinomial(const Tensor& probs, int num_samples, bool replacement) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    int num_classes = probs.shape().back();
    int num_distributions = probs.size() / num_classes;
    
    if(!replacement && num_samples > num_classes) {
        throw std::runtime_error("cannot sample more than num_classes without replacement");
    }
    
    vector<int> result_shape = probs.shape();
    result_shape.back() = num_samples;
    Tensor result(result_shape, 0.0f, false, probs.device());
    
    for(int d = 0; d < num_distributions; d++) {
        vector<float> weights(probs.data() + d * num_classes, probs.data() + (d + 1) * num_classes);
        
        for(int s = 0; s < num_samples; s++) {
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            int idx = dist(gen);
            if(!replacement) weights[idx] = 0.0f;
            result.at(d * num_samples + s) = (float)idx;
        }
    }
    
    return result;
}

Tensor randint(int low, int high, const vector<int>& shape, DeviceType device) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(low, high - 1);
    
    Tensor result(shape, 0.0f, false, device);
    for(int i = 0; i < result.size(); i++) {
        result.at(i) = (float)dist(gen);
    }
    return result;
}

Tensor layer_norm(const Tensor& A, const Tensor& gamma, const Tensor& beta, float epsilon) {
    int axis = A.shape().size() - 1;
    int norm_size = A.shape()[axis];
    
    if(gamma.size() != norm_size || beta.size() != norm_size) {
        throw std::runtime_error("gamma and beta must match last dimension size");
    }
    
    Tensor mean = A.mean(axis, true);
    Tensor var = A.variance_squared(axis, true);
    Tensor eps(var.shape(), epsilon, false, A.device());
    Tensor std_inv = (var + eps).pow(-0.5f);
    Tensor x_norm = (A - mean.broadcast(A.shape())) * std_inv.broadcast(A.shape());
    Tensor result = x_norm * gamma.broadcast(A.shape()) + beta.broadcast(A.shape());
    
    return result;
}

Tensor embedding(const Tensor& weight, const Tensor& indices) {
    if(weight.shape().size() != 2) {
        throw std::runtime_error("Embedding weight must be 2D (vocab_size x embedding_dim)");
    }
    
    int vocab_size = weight.shape()[0];
    int embed_dim = weight.shape()[1];
    
    for(int i = 0; i < indices.size(); i++) {
        int idx = (int)indices.at(i);
        if(idx < 0 || idx >= vocab_size) {
            throw std::runtime_error("Embedding index out of range");
        }
    }
    
    vector<int> result_shape = indices.shape();
    result_shape.push_back(embed_dim);
    Tensor result(result_shape, weight.requires_grad(), weight.device());
    
    for(int idx = 0; idx < result.size(); idx++) {
        int i = idx / embed_dim;
        int j = idx % embed_dim;
        int row = (int)indices.at(i);
        result.at(idx) = weight.at(row * embed_dim + j);
    }
    
    if(weight.requires_grad()) {
        shared_ptr<TensorImpl> weight_impl = weight.impl;
        shared_ptr<TensorImpl> result_impl = result.impl;
        shared_ptr<TensorImpl> indices_impl = indices.impl;

        result.impl->parents.push_back(weight_impl);
        result.impl->backward_fn = [weight_impl, result_impl, indices_impl, embed_dim]() {
            Tensor weight(weight_impl);
            Tensor result(result_impl);
            Tensor indices(indices_impl);
            
            for(int idx = 0; idx < result.size(); idx++) {
                int i = idx / embed_dim;
                int j = idx % embed_dim;
                int row = (int)indices.at(i);
                weight.grad().at(row * embed_dim + j) += result.grad().at(idx);
            }
        };
    }
    
    return result;
}

Tensor embedding(const Tensor& weight, const vector<int>& indices) {
    // Convert vector<int> to Tensor and call main implementation
    vector<float> indices_float(indices.begin(), indices.end());
    Tensor indices_tensor({(int)indices.size()}, indices_float, false, weight.device());
    return embedding(weight, indices_tensor);
}

