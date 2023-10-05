#include <iostream>

#define N 32

__global__ void ArrayAdd(int* a, int* b, int* c) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // 分配CPU内存
    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];

    // 生成输入数据
    for (int i = 0; i < N; ++i) {
        a[i] = 3 * i;
        b[i] = i * i;
    }

    // 位于CPU内存上的指向GPU内存地址的指针
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    // 分配GPU内存
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // 从CPU内存到GPU内存拷贝输入数据
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevide);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevide);

    // 执行GPU程序
    int blockDim = 32;
    int gridDim = static_cast<int>((N + (blockDim - 1)) / blockDim);
    ArrayAdd<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c);

    // 从GPU内存到CPU内存拷贝输出数据
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d", a[i], b[i], c[i];
    }

    // 释放分配的GPU内存
    cudaFree(dev_a); dev_a = nullptr;
    cudaFree(dev_b); dev_b = nullptr;
    cudaFree(dev_c); dev_c = nullptr;

    delete [] a; a = nullptr;
    delete [] b; b = nullptr;
    delete [] c; c = nullptr;

    return 0;
}