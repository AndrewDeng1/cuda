#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from the GPU!\n");
}

int main() {
    std::cout << "Hello from the CPU!" << std::endl;

    // Launch kernel with 1 block of 1 thread
    helloFromGPU<<<1, 1>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
