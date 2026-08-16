#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
// threads = BM*BN/(TM*TN) = 256

__global__ void __launch_bounds__(256)
matmul_2d(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int threadRow = threadIdx.x / (BN / TN);          // TODO(0a) 自己推:tid=37 → tRow=2
    const int threadCol = (threadIdx.x % (BN / TN)) * TN;   // tCol=40

    const int globalRow = blockIdx.y * BM + threadRow * TM; 
    const int globalCol = blockIdx.x * BN + threadCol; 

    float acc[TM][TN];               // 64 个累加器
    for (int r = 0; r < TM; ++r)
        for (int c = 0; c < TN; ++c)
            acc[r][c] = 0.f;

    for (int bk = 0; bk < K; bk += BK) {
        // TODO(1) 跨步搬运:As 装 BM*BK 个、Bs 装 BK*BN 个,每人 4+4 个
        //   As[i/BK][i%BK] ← A[(blockIdx.y*BM + i/BK)*K + bk + i%BK]
        //   Bs[i/BN][i%BN] ← B[(bk + i/BN)*N + blockIdx.x*BN + i%BN]
        for (int i = threadIdx.x; i < BM*BK; i += blockDim.x)
            As[i/BK][i%BK] = A[(blockIdx.y*BM + i/BK)*K + bk + i%BK];
        for (int i = threadIdx.x; i < BK*BN; i += blockDim.x)
            Bs[i/BN][i%BN] = B[(bk + i/BN)*N + blockIdx.x*BN + i%BN];
        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            // TODO(2) 今天的灵魂,三步:
            float a[TM];
            for (int r = 0; r < TM; ++r) a[r] = As[threadRow*TM + r][kk];   // ← 加 *TM
            float b[TN];
            for (int c = 0; c < TN; ++c) b[c] = Bs[kk][threadCol + c];
            for (int r = 0; r < TM; ++r)
                for (int c = 0; c < TN; ++c)
                    acc[r][c] += a[r] * b[c];
            //   ① 取一列 a:for r: a[r] = As[threadRow + r][kk]
            //   ② 取一行 b:for c: b[c] = Bs[kk][threadCol + c]
            //   ③ 双层循环 64 条 FMA:acc[r][c] += a[r] * b[c]
        }
        __syncthreads();
    }
    // TODO(3) 二维写回:for r: for c: C[(globalRow+r)*N + globalCol+c] = acc[r][c]
    for (int r = 0; r < TM; ++r)        // 写回的界也换成 TM/TN
        for (int c = 0; c < TN; ++c)
            C[(globalRow+r)*N + globalCol+c] = acc[r][c];
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(N/BN, M/BM);

    // warmup
    matmul_2d<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(e)); return 1; }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_2d<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

    // 全矩阵校验(4096 规模用,CPU 算一遍)
    float maxErr = 0;
    for (int r = 0; r < M/4; ++r)
        for (int c = 0; c < N/4; ++c) {
            float ref = 0;
            for (int k = 0; k < K; ++k) ref += h_A[r*K+k] * h_B[k*N+c];
            float err = fabsf(h_C[r*N+c] - ref);
            if (err > maxErr) maxErr = err;
        }
    printf("max|err| = %e %s\n", maxErr, maxErr < 1e-3f ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}