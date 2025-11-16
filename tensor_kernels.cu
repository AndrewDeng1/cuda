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
    bool cpu;
    
    TensorStruct(){}

    // Constructor that takes a tensor
    TensorStruct(bool t_cpu=true){
        cpu=t_cpu;
    }

    TensorStruct(shared_ptr<Tensor> t, bool t_cpu=true) {
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
        cpu=t_cpu;
    }

    // Constructor that takes a tensor
    TensorStruct(float* t_data, int t_data_size, vector<int>& t_shape, vector<int>& t_strides, bool t_cpu=true) {
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
        cpu=t_cpu;
    }
    
    // Destructor to free memory
    ~TensorStruct() {
        if(!cpu){
            return;
        }
        printf("Destructor called\n");
        delete[] shape;
        printf("Deleted shape\n");
        delete[] strides;
        printf("Deleted strides\n");
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

void cudaMallocTensorStruct(TensorStruct& a, TensorStruct& b){
    cudaMalloc(&a.shape, b.shape_size_bytes);
    cudaMalloc(&a.strides, b.strides_size_bytes);
    cudaMalloc(&a.data, b.data_size_bytes);
}

void cudaMemcpyTensorStruct(TensorStruct& targ, TensorStruct& src, cudaMemcpyKind dir){
    cudaMemcpy(targ.shape, src.shape, src.shape_size_bytes, dir);
    cudaMemcpy(targ.strides, src.strides, src.strides_size_bytes, dir);
    cudaMemcpy(targ.data, src.data, src.data_size_bytes, dir);
}

void cudaFreeTensorStruct(TensorStruct t){
    cudaFree(t.data);
    cudaFree(t.shape);
    cudaFree(t.strides);
}

__global__ void addKernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]+b[idx];
    }
}

void launchAdd(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    addKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void subtractKernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]-b[idx];
    }
}

void launchSubtract(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    subtractKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void multiplyKernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]*b[idx];
    }
}

void launchMultiply(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    multiplyKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void divideKernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]/b[idx];
    }
}

void launchDivide(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    divideKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// __global__ void broadcastKernel(float* a, float* b, int N){
// // __global__ void broadcastKernel(TensorStruct a, TensorStruct b, bool matmul){
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     // printf("idx: %d\n", idx);

//     if(idx>=N)return;

//     b[idx] = 1.0;
//     // int curr = idx;
//     // int cnt = 0;
//     // // printf("(%d) start for. b.shape_size: %d, matmul: %d\n", idx, b.shape_size, matmul);
//     // for(int j=0; j<b.shape_size-2*matmul; j++) {
//     //     // printf("(%d) iter\n", idx);
//     //     // printf("(%d) b.strides: %d\n", idx, b.strides);
//     //     // printf("(%d) b.strides[j]: %d\n", idx, b.strides[j]);
//     //     int dim = curr/b.strides[j];
//     //     // printf("(%d) dim: %d\n", idx, dim);
//     //     curr %= b.strides[j];
//     //     // printf("(%d) part a, a.shape_size: %d, j: %d\n", idx, a.shape_size, j);
//     //     // printf("a.shape\n");
//     //     // printf("a.shape[0]=%d\n", a.shape[0]);
//     //     // for(int x=0; x<a.shape_size; x++){
//     //     //     // printf("(%d) hi\n", idx);
//     //     //     printf("(%d) a.shape[%d] = %d\n", idx, x, a.shape[x]);
//     //     // }
//     //     if(a.shape[j] == 1) {
//     //         printf("(%d) part b\n", idx);
//     //         cnt += 0;  // Don't add to index for broadcasted dimensions
//     //     } else {
//     //         printf("(%d) part c\n", idx);
//     //         cnt += a.strides[j] * dim;
//     //     }
//     //     printf("(%d) done inner\n", idx);
//     // }
//     // printf("(%d) cnt=%d, a.data[cnt]=%f\n", idx, cnt, a.data[cnt]);
//     // b.data[idx] = a.data[cnt];
//     // printf("(%d) done, b.data[%d]=%f\n", idx, idx, b.data[idx]);
// }
__global__ void broadcastKernel(
    int a_shape_size, 
    int* a_shape,
    int a_strides_size,
    int* a_strides,
    int a_data_size, 
    float* a_data, 
    int b_shape_size, 
    int* b_shape,
    int b_strides_size,
    int* b_strides,
    int b_data_size, 
    float* b_data, 
    bool matmul
){
// __global__ void broadcastKernel(TensorStruct a, TensorStruct b, bool matmul){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("idx: %d\n", idx);

    if(idx>=b_data_size)return;

    int curr = idx;
    int cnt = 0;
    // printf("(%d) start for. b.shape_size: %d, matmul: %d\n", idx, b.shape_size, matmul);
    for(int j=0; j<b_shape_size-2*matmul; j++) {
        // printf("(%d) iter\n", idx);
        // printf("(%d) b.strides: %d\n", idx, b.strides);
        // printf("(%d) b.strides[j]: %d\n", idx, b.strides[j]);
        int dim = curr/b_strides[j];
        // printf("(%d) dim: %d\n", idx, dim);
        curr %= b_strides[j];
        // printf("(%d) part a, a.shape_size: %d, j: %d\n", idx, a.shape_size, j);
        // printf("a.shape\n");
        // printf("a.shape[0]=%d\n", a.shape[0]);
        // for(int x=0; x<a.shape_size; x++){
        //     // printf("(%d) hi\n", idx);
        //     printf("(%d) a.shape[%d] = %d\n", idx, x, a.shape[x]);
        // }
        if(a_shape[j] == 1) {
            cnt += 0;  // Don't add to index for broadcasted dimensions
        } else {
            cnt += a_strides[j] * dim;
        }
    }
    b_data[idx] = a_data[cnt];
}

void launchBroadcast(shared_ptr<Tensor>a, shared_ptr<Tensor>b, vector<int>& padded_shape, vector<int>& padded_strides, bool matmul){
    TensorStruct a_struct(a->data, a->size(), padded_shape, padded_strides);
    TensorStruct b_struct(b);
    int N = b->size();
    printf("a");
    TensorStruct d_a_struct(false);
    TensorStruct d_b_struct(false);
    printf("b");
    
    // cudaMallocTensorStruct(d_a_struct, a_struct);
    // cudaMallocTensorStruct(d_b_struct, b_struct);
    
    // Manual allocation for d_a_struct
    cudaMalloc(&d_a_struct.shape, a_struct.shape_size_bytes);
    cudaMalloc(&d_a_struct.strides, a_struct.strides_size_bytes);
    cudaMalloc(&d_a_struct.data, a_struct.data_size_bytes);
    
    // Manual allocation for d_b_struct
    cudaMalloc(&d_b_struct.shape, b_struct.shape_size_bytes);
    cudaMalloc(&d_b_struct.strides, b_struct.strides_size_bytes);
    cudaMalloc(&d_b_struct.data, b_struct.data_size_bytes);
    printf("c");
    
    // Populate scalar fields in device-pointer structs (passed by value to kernel)
    d_a_struct.shape_size = a_struct.shape_size;
    d_a_struct.shape_size_bytes = a_struct.shape_size_bytes;
    d_a_struct.strides_size = a_struct.strides_size;
    d_a_struct.strides_size_bytes = a_struct.strides_size_bytes;
    d_a_struct.data_size = a_struct.data_size;
    d_a_struct.data_size_bytes = a_struct.data_size_bytes;

    d_b_struct.shape_size = b_struct.shape_size;
    d_b_struct.shape_size_bytes = b_struct.shape_size_bytes;
    d_b_struct.strides_size = b_struct.strides_size;
    d_b_struct.strides_size_bytes = b_struct.strides_size_bytes;
    d_b_struct.data_size = b_struct.data_size;
    d_b_struct.data_size_bytes = b_struct.data_size_bytes;
    
    
    // cudaMemcpyTensorStruct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    // cudaMemcpyTensorStruct(d_b_struct, b_struct, cudaMemcpyHostToDevice);
    
    // Manual copy for a (data, shape, strides)
    cudaMemcpy(d_a_struct.data, a_struct.data, a_struct.data_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_struct.shape, a_struct.shape, a_struct.shape_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_struct.strides, a_struct.strides, a_struct.strides_size_bytes, cudaMemcpyHostToDevice);
    
    // Manual copy for b (ONLY shape and strides, NOT data)
    cudaMemcpy(d_b_struct.shape, b_struct.shape, b_struct.shape_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_struct.strides, b_struct.strides, b_struct.strides_size_bytes, cudaMemcpyHostToDevice);
    
    broadcastKernel<<<(N+255)/256, 256>>>(
        d_a_struct.shape_size,
        d_a_struct.shape,
        d_a_struct.strides_size,
        d_a_struct.strides,
        d_a_struct.data_size,
        d_a_struct.data, 
        d_b_struct.shape_size,
        d_b_struct.shape,
        d_b_struct.strides_size,
        d_b_struct.strides,
        d_b_struct.data_size,
        d_b_struct.data, 
        matmul);
    // broadcastKernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, matmul);
    printf("c2");
    cudaDeviceSynchronize();
    printf("c3");
    
    // Copy result data back from device to host
    cudaMemcpy(b_struct.data, d_b_struct.data, b_struct.data_size_bytes, cudaMemcpyDeviceToHost);
    printf("d");
    
    printf("e");
    // cudaFreeTensorStruct(d_a_struct);
    // cudaFreeTensorStruct(d_b_struct);
    
    // Manual cleanup for d_a_struct
    cudaFree(d_a_struct.data);
    cudaFree(d_a_struct.shape);
    cudaFree(d_a_struct.strides);
    
    // Manual cleanup for d_b_struct
    cudaFree(d_b_struct.data);
    cudaFree(d_b_struct.shape);
    cudaFree(d_b_struct.strides);
    printf("f");
}