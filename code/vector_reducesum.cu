#include <iostream>

__global__ void reduceCompleteUnrollWarp8(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    if (tid >= n) return;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if(idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    //in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512) {
        idata[tid] += idata[tid + 512];
    }
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) {
        idata[tid] += idata[tid + 256];
    }
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) {
        idata[tid] += idata[tid + 128];
    }
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) {
        idata[tid] += idata[tid + 64];
    }
    __syncthreads();

    //write result for this block to global mem
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnroll2(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    if (tid >= n) return;
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // 这一句是核心，添加来自相邻数据块的值。
    if (idx + blockDim.x < n) {
        g_idata[idx] += g_idata[idx + blockDim.x];
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    //write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) return;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) return;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) return;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main (int argc, char** argv) {
    // initialization
    int N = 1 << 24;

    // execution configuration
    dim3 block(1024);
    dim3 grid((N - 1) / block.x + 1);

    // allocate host memory
    int* idata_host = new int[N];
    int* odata_host = new int[grid.x];

    // initialize the array
    for (int i = 0; i < N; i++) {
        idata_host[i] = i;
    }

    // device memory
    int* idata_dev = nullptr;
    int* odata_dev = nullptr;
    CHECK(cudaMalloc((void**)&idata_dev, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

    // kernel reduceNeighbored
    CHECK(cudaMemcpy(idata_dev, idata_host, N * sizeof(int), cudaMemcpyHostToDevice));

    reduceNeighbored <<<grid, block >>>(idata_dev, odata_dev, size);

    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    
    // free host memory
    free(idata_host); idata_host = nullptr;
    free(odata_host); odata_host = nullptr;
    CHECK(cudaFree(idata_dev)); idata_dev = nullptr;
    CHECK(cudaFree(odata_dev)); odata_dev = nullptr;

    return EXIT_SUCCESS;
}