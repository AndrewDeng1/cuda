#include<iostream>
using namespace std;

__global__ void addKernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]+b[idx];
    }
}

void launchAdd(float* a, float* b, float* c, int N){
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    addKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

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

void launchSubtract(float* a, float* b, float* c, int N){
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    subtractKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

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

void launchMultiply(float* a, float* b, float* c, int N){
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    multiplyKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

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

void launchDivide(float* a, float* b, float* c, int N){
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    divideKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}