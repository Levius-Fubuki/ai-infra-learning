// matmul_tiled.cu —— Week2 Day5: shared memory tiling 优化
// 编译: nvcc matmul_tiled.cu -o matmul_tiled && ./matmul_tiled

#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

const int TILE = 16;   // 分块大小 16×16

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[16][16];   // A 的一块
    __shared__ float Bs[16][16];   // B 的一块

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // K 维分阶段,每阶段搬一个 16×16 块
    for (int ph = 0; ph < (K + TILE - 1) / TILE; ++ph) { //64个ph
        // ① 合作搬运:每线程搬 As 的 1 个元素
        if (row < M && ph * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + ph * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        // 搬 Bs 的 1 个元素
        if (ph * TILE + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(ph * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();   // ⭐ 搬完才能算

        // ② 在 shared 上算 16 次乘加(快!不碰全局内存)
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();   // ⭐ 算完才能进下一轮搬运(防 As/Bs 被覆盖)
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    // warmup
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms / 1000.0) / 1e12;
    cout << "tiled matmul " << M << "x" << N << "x" << K << "\n";
    cout << "耗时 = " << ms << " ms\n";
    cout << "算力 = " << tflops << " TFLOPS\n";
    cout << "3090 峰值 ~35 TFLOPS → 利用率 " << tflops / 35.0 * 100 << " %\n";
    cout << "对比 naive: 加速 " << tflops / 0.514 << " x\n";

    // 验证
    float cpu_c00 = 0;
    for (int k = 0; k < K; ++k) cpu_c00 += h_A[0 * K + k] * h_B[k * N + 0];
    cout << "验证 C[0][0]: GPU=" << h_C[0] << "  CPU=" << cpu_c00 << "\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}