#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

// If compiling without CUDA
#ifndef __CUDACC__
#include <functional>
#endif

using namespace std;

// std::function isn't available in CUDA device code, so backward_fn needs special behavior
#ifdef __CUDACC__
// For CUDA compilation, use a placeholder type
// backward_fn is never accessed in CUDA code
struct BackwardFnPlaceholder {
    char dummy;
};
typedef BackwardFnPlaceholder BackwardFnType;
#else
typedef function<void()> BackwardFnType;
#endif

enum class DeviceType {
    CPU,
    CUDA
};

// Forward declarations
class Tensor;
class TensorImpl;

// TODO: Revert default device to CUDA
// TensorImpl contains all the actual data and fields
class TensorImpl : public enable_shared_from_this<TensorImpl> {
public:
    // Constructors
    TensorImpl();
    TensorImpl(const vector<int>& shape, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    TensorImpl(const vector<int>& shape, const vector<float>& data, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    TensorImpl(const vector<int>& shape, float* data, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    TensorImpl(const vector<int>& shape, float num, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    TensorImpl(shared_ptr<TensorImpl> other);

    // Big 5
    ~TensorImpl();                                      // Destructor
    TensorImpl(const TensorImpl& other);                // Copy constructor
    TensorImpl& operator=(const TensorImpl& other);     // Copy assignment
    TensorImpl(TensorImpl&& other) noexcept;            // Move constructor
    TensorImpl& operator=(TensorImpl&& other) noexcept; // Move assignment

    int size() const;
    float& at(vector<int> indices);
    float& at(int index);
    vector<int> compute_strides(const vector<int>& shape);
    void print();

    // Member variables
    bool requires_grad;
    DeviceType device;
    vector<int> shape;
    vector<int> strides;
    float* data;
    shared_ptr<TensorImpl> grad;  // grad is now TensorImpl
    vector<shared_ptr<TensorImpl>> parents;  // parents are TensorImpl
    BackwardFnType backward_fn;
};

// Tensor is a wrapper around TensorImpl
class Tensor {
public:
    shared_ptr<TensorImpl> impl;

    // Constructors
    Tensor();
    Tensor(shared_ptr<TensorImpl> impl);  // Wrap existing impl
    Tensor(const vector<int>& shape, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    Tensor(const vector<int>& shape, const vector<float>& data, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    Tensor(const vector<int>& shape, float* data, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    Tensor(const vector<int>& shape, float num, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    // Initializer list constructors for cleaner syntax
    Tensor(initializer_list<int> shape, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    Tensor(initializer_list<int> shape, initializer_list<float> data, bool requires_grad=false, DeviceType device=DeviceType::CUDA);
    Tensor(initializer_list<int> shape, float num, bool requires_grad=false, DeviceType device=DeviceType::CUDA);

    // Big 5
    ~Tensor() = default;                            // Destructor (shared_ptr handles cleanup)
    Tensor(const Tensor& other);                    // Copy constructor (deep copy)
    Tensor& operator=(const Tensor& other);         // Copy assignment (deep copy)
    Tensor(Tensor&& other) noexcept;                // Move constructor
    Tensor& operator=(Tensor&& other) noexcept;     // Move assignment
    
    Tensor clone() const;                           // Deep copy

    // Accessors (forward to impl)
    int size() const;
    float& at(vector<int> indices);
    float& at(int index);
    float& at(vector<int> indices) const;
    float& at(int index) const;
    vector<int> compute_strides(const vector<int>& shape);
    
    // Member functions
    Tensor reshape(const vector<int>& new_shape) const;
    Tensor reduce_to_shape(const vector<int>& target_shape) const;
    Tensor sum(int axis, bool keepdims=true) const;
    Tensor transpose(int dim1, int dim2) const;
    Tensor mean(int axis, bool keepdims=true) const;
    Tensor variance_squared(int axis, bool keepdims=true) const;
    Tensor pow(float exponent) const;
    Tensor slice(int dim, int start, int end) const;
    Tensor masked_fill(const Tensor& mask, float value) const;
    void backward();
    void print() const;

    // Broadcasting and reduction operations
    Tensor broadcast(const vector<int>& new_shape, bool matmul = false) const;

    // Property accessors
    bool requires_grad() const { return impl->requires_grad; }
    void set_requires_grad(bool val);
    DeviceType device() const { return impl->device; }
    const vector<int>& shape() const { return impl->shape; }
    const vector<int>& strides() const { return impl->strides; }
    float* data() const { return impl->data; }
    
    // Grad accessor - returns Tensor wrapping the grad impl
    Tensor grad() const;
    void set_grad(const Tensor& g);
    bool has_grad() const { return impl->grad != nullptr; }
};

// Free function declarations - now take Tensor& instead of shared_ptr<Tensor>
Tensor operator+(const Tensor& A, const Tensor& B);
Tensor operator-(const Tensor& A, const Tensor& B);
Tensor operator*(const Tensor& A, const Tensor& B);
Tensor operator/(const Tensor& A, const Tensor& B);
Tensor& operator+=(Tensor& A, const Tensor& B);
Tensor& operator-=(Tensor& A, const Tensor& B);
Tensor& operator*=(Tensor& A, const Tensor& B);
Tensor& operator/=(Tensor& A, const Tensor& B);

// Scalar-tensor operators
Tensor operator+(const Tensor& A, float B);
Tensor operator+(float A, const Tensor& B);
Tensor operator-(const Tensor& A, float B);
Tensor operator-(float A, const Tensor& B);
Tensor operator-(const Tensor& A);
Tensor operator*(const Tensor& A, float B);
Tensor operator*(float A, const Tensor& B);
Tensor operator/(const Tensor& A, float B);
Tensor operator/(float A, const Tensor& B);

Tensor matmul(const Tensor& A, const Tensor& B);

Tensor relu(const Tensor& A);
Tensor sigmoid(const Tensor& A);
Tensor tanh_op(const Tensor& A);  // renamed to avoid conflict with std::tanh
Tensor softmax(const Tensor& A, int axis);
Tensor dropout(const Tensor& A, float p = 0.5f, bool training = true);
Tensor cat(const vector<Tensor>& tensors, int axis);
Tensor stack(const vector<Tensor>& tensors, int axis = 0);
Tensor layer_norm(const Tensor& A, const Tensor& gamma, const Tensor& beta, float epsilon = 1e-5f);
Tensor embedding(const Tensor& weight, const Tensor& indices);
Tensor embedding(const Tensor& weight, const vector<int>& indices);
Tensor tril(int rows, int cols, DeviceType device = DeviceType::CUDA);
Tensor arange(float start, float end, float step = 1.0f, DeviceType device = DeviceType::CUDA);
Tensor multinomial(const Tensor& probs, int num_samples, bool replacement = false);
Tensor randint(int low, int high, const vector<int>& shape, DeviceType device = DeviceType::CUDA);
Tensor randn(const vector<int>& shape, DeviceType device = DeviceType::CUDA);  // N(0,1)
Tensor xavier_normal(const vector<int>& shape, DeviceType device = DeviceType::CUDA);   // for tanh/sigmoid/transformers
Tensor kaiming_normal(const vector<int>& shape, DeviceType device = DeviceType::CUDA);  // for ReLU
Tensor zeros(const vector<int>& shape, DeviceType device = DeviceType::CUDA);
Tensor ones(const vector<int>& shape, DeviceType device = DeviceType::CUDA);
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);  // logits: (N, C) or (..., C), targets: (N,) or (...)

// Global functions
bool is_broadcastable(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul = false);
vector<int> get_broadcast_shape(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul = false);

#endif // TENSOR_H
