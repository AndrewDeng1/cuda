#include<iostream>
#include<vector>
#include<cfloat>
#include "tensor.h"
using namespace std;

struct TensorStruct {
    int* shape;
    int shape_size;
    size_t shape_size_bytes;
    int* strides;
    int strides_size;
    size_t strides_size_bytes;
    float* data;
    int data_size;
    size_t data_size_bytes;
    bool on_cpu;
    
    TensorStruct(){}

    // Constructor that takes a bool
    TensorStruct(bool t_on_cpu=false){
        on_cpu=t_on_cpu;
    }

    TensorStruct(shared_ptr<Tensor> t, bool t_on_cpu=false) {
        shape_size = t->shape.size();
        shape_size_bytes = shape_size*sizeof(int);
        shape = new int[shape_size];
        for(int i = 0; i < shape_size; i++) {
            shape[i] = t->shape[i];
        }
        
        strides_size = t->strides.size();
        strides_size_bytes = strides_size*sizeof(int);
        strides = new int[strides_size];
        for(int i = 0; i < strides_size; i++) {
            strides[i] = t->strides[i];
        }
        
        data = t->data;
        data_size = t->size();
        data_size_bytes = data_size*sizeof(float);
        on_cpu=t_on_cpu;
    }

    // Constructor that takes raw data
    TensorStruct(float* t_data, int t_data_size, vector<int>& t_shape, vector<int>& t_strides, bool t_on_cpu=false) {
        shape_size = t_shape.size();
        shape_size_bytes = shape_size*sizeof(int);
        shape = new int[shape_size];
        for(int i = 0; i < shape_size; i++) {
            shape[i] = t_shape[i];
        }
        
        strides_size = t_strides.size();
        strides_size_bytes = strides_size*sizeof(int);
        strides = new int[strides_size];
        for(int i = 0; i < strides_size; i++) {
            strides[i] = t_strides[i];
        }
        
        data = t_data;
        data_size = t_data_size;
        data_size_bytes = data_size*sizeof(float);
        on_cpu=t_on_cpu;
    }
    
    // Destructor to free memory
    ~TensorStruct() {

        // If TensorStruct holds memory addresses on gpu, don't free memory (it's freed via cudaFree)
        if(!on_cpu){
            return;
        }

        printf("detor working\n");

        // Don't free data field because that holds same address as data field in original tensor
        delete[] shape;
        delete[] strides;
    }

    // Unnecessary as long as the shape and strides for the tensor are unaffected by operation,
    // assuming you're doing standard 1. Create TensorStruct 2. cudaMallocTensorStruct
    // 3. cudaMemcpyTensorStruct 4. Operation 5. cudaMemcpyTensorSTruct back 6. cudaFreeTensorStruct
    void toTensor(shared_ptr<Tensor> t){
        t->data = data;
        t->shape = vector<int>();
        for(int i=0; i<shape_size; i++){
            t->shape.push_back(shape[i]);
        }
        t->strides = vector<int>();
        for(int i=0; i<strides_size; i++){
            t->strides.push_back(strides[i]);
        }
    }
};

void cuda_malloc_tensor_struct(TensorStruct& a, TensorStruct& b){
    cudaMalloc(&a.shape, b.shape_size_bytes);
    cudaMalloc(&a.strides, b.strides_size_bytes);
    cudaMalloc(&a.data, b.data_size_bytes);
}

void cuda_memcpy_tensor_struct(TensorStruct& targ, TensorStruct& src, cudaMemcpyKind dir){
    cudaMemcpy(targ.shape, src.shape, src.shape_size_bytes, dir);
    cudaMemcpy(targ.strides, src.strides, src.strides_size_bytes, dir);
    cudaMemcpy(targ.data, src.data, src.data_size_bytes, dir);

    // Copy scalar fields from b to a
    targ.shape_size = src.shape_size;
    targ.shape_size_bytes = src.shape_size_bytes;
    targ.strides_size = src.strides_size;
    targ.strides_size_bytes = src.strides_size_bytes;
    targ.data_size = src.data_size;
    targ.data_size_bytes = src.data_size_bytes;
}

void cuda_free_tensor_struct(TensorStruct t){
    cudaFree(t.data);
    cudaFree(t.shape);
    cudaFree(t.strides);
}

__global__ void add_kernel(TensorStruct a, TensorStruct b, TensorStruct c){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= c.data_size) return;
    c.data[idx] = a.data[idx] + b.data[idx];
}

void launch_add(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    TensorStruct c_struct(result);
    int N = result->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    TensorStruct d_c_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    cuda_malloc_tensor_struct(d_c_struct, c_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_c_struct, c_struct, cudaMemcpyHostToDevice);

    add_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, d_c_struct);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(c_struct, d_c_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
    cuda_free_tensor_struct(d_c_struct);
}

__global__ void subtract_kernel(TensorStruct a, TensorStruct b, TensorStruct c){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= c.data_size) return;
    c.data[idx] = a.data[idx] - b.data[idx];
}

void launch_subtract(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    TensorStruct c_struct(result);
    int N = result->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    TensorStruct d_c_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    cuda_malloc_tensor_struct(d_c_struct, c_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_c_struct, c_struct, cudaMemcpyHostToDevice);

    subtract_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, d_c_struct);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(c_struct, d_c_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
    cuda_free_tensor_struct(d_c_struct);
}

__global__ void multiply_kernel(TensorStruct a, TensorStruct b, TensorStruct c){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= c.data_size) return;
    c.data[idx] = a.data[idx] * b.data[idx];
}

void launch_multiply(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    TensorStruct c_struct(result);
    int N = result->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    TensorStruct d_c_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    cuda_malloc_tensor_struct(d_c_struct, c_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_c_struct, c_struct, cudaMemcpyHostToDevice);

    multiply_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, d_c_struct);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(c_struct, d_c_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
    cuda_free_tensor_struct(d_c_struct);
}

__global__ void divide_kernel(TensorStruct a, TensorStruct b, TensorStruct c){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= c.data_size) return;
    c.data[idx] = a.data[idx] / b.data[idx];
}

void launch_divide(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    TensorStruct c_struct(result);
    int N = result->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    TensorStruct d_c_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    cuda_malloc_tensor_struct(d_c_struct, c_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_c_struct, c_struct, cudaMemcpyHostToDevice);

    divide_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, d_c_struct);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(c_struct, d_c_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
    cuda_free_tensor_struct(d_c_struct);
}

__global__ void broadcast_kernel(TensorStruct a, TensorStruct b, bool matmul){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx>=b.data_size) return;

    int curr = idx;
    int cnt = 0;
    for(int j=0; j<b.shape_size-2*matmul; j++) {
        int dim = curr/b.strides[j];
        curr %= b.strides[j];
        if(a.shape[j] == 1) {
            cnt += 0;  // Don't add to index for broadcasted dimensions
        } else {
            cnt += a.strides[j] * dim;
        }
    }
    b.data[idx] = a.data[cnt];
}

void launch_broadcast(shared_ptr<Tensor>a, shared_ptr<Tensor>b, vector<int>& padded_shape, vector<int>& padded_strides, bool matmul){
    TensorStruct a_struct(a->data, a->size(), padded_shape, padded_strides);
    TensorStruct b_struct(b);
    int N = b->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);

    broadcast_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, matmul);
    cudaDeviceSynchronize();
    
    // Copy result data back from device to host
    cuda_memcpy_tensor_struct(b_struct, d_b_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
}

__global__ void sum_kernel(TensorStruct a, TensorStruct b, int axis){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx>=b.data_size) return;

    for(int i=0; i<a.shape[axis]; i++){

        int curr=idx;
        int j=0;
        for(int x=0; x<a.shape_size; x++){
            if(x==axis){
                j+=i*a.strides[x];
            } else {
                j+=(curr/b.strides[x])*a.strides[x];
            }
            curr%=b.strides[x];
        }

        b.data[idx]+=a.data[j];
    }
}

void launch_sum(shared_ptr<Tensor>a, shared_ptr<Tensor>b, int axis){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    int N = b->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);

    sum_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, axis);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(b_struct, d_b_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
}

__global__ void transpose_kernel(TensorStruct a, TensorStruct b, int dim1, int dim2){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx>=b.data_size) return;

    int ind=0;
    int curr=idx;
    int mag_dim1;
    int mag_dim2;
    for(int j=0; j<b.shape_size; j++){
        if(j==dim1){
            mag_dim1=curr/b.strides[j];
        } else if(j==dim2){
            mag_dim2=curr/b.strides[j];
        }
        ind+=(curr/b.strides[j])*a.strides[j];
        curr%=b.strides[j];
    }
    ind-=mag_dim1*a.strides[dim1];
    ind+=mag_dim2*a.strides[dim1];
    ind+=mag_dim1*a.strides[dim2];
    ind-=mag_dim2*a.strides[dim2];
    b.data[idx]=a.data[ind];
}

void launch_transpose(shared_ptr<Tensor>a, shared_ptr<Tensor>b, int dim1, int dim2){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    int N = b->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);

    transpose_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, dim1, dim2);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(b_struct, d_b_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
}

__global__ void pow_kernel(TensorStruct a, TensorStruct b, int exponent){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx>=b.data_size) return;

    b.data[idx] = powf(a.data[idx], exponent);
}

void launch_pow(shared_ptr<Tensor>a, shared_ptr<Tensor>b, int exponent){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    int N = b->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);

    pow_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, exponent);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(b_struct, d_b_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
}

__global__ void relu_kernel(TensorStruct a, TensorStruct b){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx>=b.data_size) return;

    b.data[idx] = fmaxf(0.0f, a.data[idx]);
}

void launch_relu(shared_ptr<Tensor>a, shared_ptr<Tensor>b){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    int N = b->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);

    relu_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(b_struct, d_b_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
}

__global__ void sigmoid_kernel(TensorStruct a, TensorStruct b){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx>=b.data_size) return;

    b.data[idx] = 1.0f/(1.0f+expf(-a.data[idx]));
}

void launch_sigmoid(shared_ptr<Tensor>a, shared_ptr<Tensor>b){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    int N = b->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);

    sigmoid_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(b_struct, d_b_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
}

__global__ void tanh_kernel(TensorStruct a, TensorStruct b){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx>=b.data_size) return;

    b.data[idx] = (expf(a.data[idx]) - expf(-a.data[idx])) / (expf(a.data[idx]) + expf(-a.data[idx]));
}

void launch_tanh(shared_ptr<Tensor>a, shared_ptr<Tensor>b){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    int N = b->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);

    tanh_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(b_struct, d_b_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
}

__global__ void sum_exp_kernel(TensorStruct a, TensorStruct b, int axis, float shift = 0.0f){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx>=b.data_size) return;

    for(int j=0; j<a.shape[axis]; j++){

        int curr=idx;
        int a_idx=0;
        for(int x=0; x<a.shape_size; x++){
            if(x==axis){
                a_idx+=j*a.strides[x];
            } else {
                a_idx+=(curr/b.strides[x])*a.strides[x];
            }
            curr%=b.strides[x];
        }

        b.data[idx]+=expf(a.data[a_idx] - shift);
    }
}

__global__ void exp_kernel(TensorStruct a, TensorStruct b, float shift = 0.0f){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx>=b.data_size) return;

    b.data[idx]=expf(a.data[idx] - shift);
}

void launch_softmax(shared_ptr<Tensor>a, shared_ptr<Tensor>sm_exp, shared_ptr<Tensor> sm_exp_broadcast, shared_ptr<Tensor>b, int axis){
    TensorStruct a_struct(a);
    TensorStruct sm_exp_struct(sm_exp);
    TensorStruct sm_exp_broadcast_struct(sm_exp_broadcast);
    TensorStruct b_struct(b);
    int N = b->size();

    // TODO: When later make it possible to specify dtype, will have to change this code to get max value for that dtype
    // TODO: Also technically should be finding the max value along each axis and shifting by the max value per axis, this is very lazy way but works ig?
    // Find max value in input tensor for numerical stability
    float max_val = -FLT_MAX;
    for(int i = 0; i < a->size(); i++){
        if(a->data[i] > max_val){
            max_val = a->data[i];
        }
    }

    TensorStruct d_a_struct(false);
    TensorStruct d_sm_exp_struct(false);
    TensorStruct d_sm_exp_broadcast_struct(false);
    TensorStruct d_b_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_sm_exp_struct, sm_exp_struct);
    cuda_malloc_tensor_struct(d_sm_exp_broadcast_struct, sm_exp_broadcast_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_sm_exp_struct, sm_exp_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_sm_exp_broadcast_struct, sm_exp_broadcast_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);

    sum_exp_kernel<<<(sm_exp->size()+255)/256, 256>>>(d_a_struct, d_sm_exp_struct, axis, max_val);
    cudaDeviceSynchronize();
    broadcast_kernel<<<(sm_exp_broadcast->size()+255)/256, 256>>>(d_sm_exp_struct, d_sm_exp_broadcast_struct, false);
    cudaDeviceSynchronize();
    exp_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_a_struct, max_val);
    cudaDeviceSynchronize();
    divide_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_sm_exp_broadcast_struct, d_b_struct);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(b_struct, d_b_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_sm_exp_struct);
    cuda_free_tensor_struct(d_sm_exp_broadcast_struct);
    cuda_free_tensor_struct(d_b_struct);
}

// TODO: Implement with tiling
__global__ void matmul_kernel(TensorStruct a, TensorStruct b, TensorStruct c){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    int ind_A = 0;
    int ind_B = 0;
    int curr_A = idx;
    int curr_B = idx;

    // Gets you to dimension before 2D
    for(int j=0; j<a.shape_size-2; j++){
        ind_A += (curr_A/c.strides[j])*a.strides[j];
        curr_A%=c.strides[j];
    }

    // Get corresponding row
    ind_A += (curr_A/c.strides[c.strides_size-2])*a.strides[a.strides_size-2];
    
    // Ignore column
    curr_A%=c.strides[c.strides_size-1];
    
    // Gets you to dimension before 2D
    for(int j=0; j<b.shape_size-2; j++){
        ind_B += (curr_B/c.strides[j])*b.strides[j];
        curr_B%=c.strides[j];
    }

    // Ignore row
    curr_B%=c.strides[c.strides_size-2];
    
    // Get corresponding column
    ind_B += (curr_B/c.strides[c.strides_size-1])*b.strides[b.strides_size-1];
    
    for(int j=0; j<a.shape[a.shape_size-1]; j++){
        c.data[idx] += a.data[ind_A]*b.data[ind_B];
        ind_A += a.strides[a.strides_size-1];
        ind_B += b.strides[b.strides_size-2];
    }
}

void launch_matmul(shared_ptr<Tensor>a, shared_ptr<Tensor>b, shared_ptr<Tensor>c){
    TensorStruct a_struct(a);
    TensorStruct b_struct(b);
    TensorStruct c_struct(c);
    int N = c->size();

    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    TensorStruct d_c_struct(false);
    
    cuda_malloc_tensor_struct(d_a_struct, a_struct);
    cuda_malloc_tensor_struct(d_b_struct, b_struct);
    cuda_malloc_tensor_struct(d_c_struct, c_struct);
    
    cuda_memcpy_tensor_struct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_b_struct, b_struct, cudaMemcpyHostToDevice);
    cuda_memcpy_tensor_struct(d_c_struct, c_struct, cudaMemcpyHostToDevice);

    matmul_kernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, d_c_struct);
    cudaDeviceSynchronize();
    
    cuda_memcpy_tensor_struct(c_struct, d_c_struct, cudaMemcpyDeviceToHost);

    cuda_free_tensor_struct(d_a_struct);
    cuda_free_tensor_struct(d_b_struct);
    cuda_free_tensor_struct(d_c_struct);
}