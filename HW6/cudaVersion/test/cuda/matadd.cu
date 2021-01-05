#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 512
#define BLOCK_SIZE 16

// GPU 的 Kernel
__global__ void MatAdd(float *A, float *B, float *C)
{
    // 根據 CUDA 模型，算出當下 thread 對應的 x 與 y
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 換算成線性的 index
    int idx = j * N + i;

    if (i < N && j < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    int i;

    // 宣告 Host 記憶體 (線性)
    h_A = (float *)malloc(N * N * sizeof(float));
    h_B = (float *)malloc(N * N * sizeof(float));
    h_C = (float *)malloc(N * N * sizeof(float));

    // 初始化 Host 的數值
    for (i = 0; i < (N * N); i++)
    {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
        h_C[i] = 0.0;
    }

    // 宣告 Device (GPU) 記憶體
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    // 將資料傳給 Device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(N / BLOCK_SIZE, N / BLOCK_SIZE);

    // 執行 MatAdd kernel
    MatAdd<<<numBlock, blockSize>>>(d_A, d_B, d_C);

    // 等待 GPU 所有 thread 完成
    cudaDeviceSynchronize();

    // 將 Device 的資料傳回給 Host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 驗證正確性
    for (i = 0; i < (N * N); i++)
    {
        if (h_C[i] != 3.0)
        {
            printf("Error:%f, idx:%d\n", h_C[i], i);
            break;
        }
    }

    printf("PASS\n");

    // free memory

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
