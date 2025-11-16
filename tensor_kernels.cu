#include<iostream>
#include<vector>
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

__global__ void add_kernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]+b[idx];
    }
}

void launch_add(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    add_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void subtract_kernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]-b[idx];
    }
}

void launch_subtract(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    subtract_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void multiply_kernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]*b[idx];
    }
}

void launch_multiply(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    multiply_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void divide_kernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]/b[idx];
    }
}

void launch_divide(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    divide_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
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