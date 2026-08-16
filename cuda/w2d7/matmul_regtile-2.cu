#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

#define BM 128
#define BN 128
#define BK 8
#define TM 8
// threads = BM*BN/TM = 512

__global__ void __launch_bounds__(512)
matmul_regtile(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];   // A 的 64×8 切片
    __shared__ float Bs[BK][BN];   // B 的 8×64 切片

    // 512 人分 64 行 → 每行 8 个线程;每人负责本行连续 8 列
    const int threadRow = threadIdx.x / (BN / TM);        // 0..63
    const int threadCol = (threadIdx.x % (BN / TM)) * TM; // 0,8,...,56

    const int globalRow = blockIdx.y * BM + threadRow;
    const int globalCol = blockIdx.x * BN + threadCol;

    float acc[TM] = {0.f};   // 8 个累加器:8 个寄存器,这 8 个数全程不出寄存器

    // 合作搬运:512 人,每人搬 1 个 A + 1 个 B(BM*BK=512,BK*BN=512,正好)
    const int loadA = threadIdx.x;          // 线性铺进 As
    const int loadB = threadIdx.x;          // 线性铺进 Bs

    for (int bk = 0; bk < K; bk += BK) {
        // A 是行主序 M×K:As[loadA/BK][loadA%BK] = A[(blockIdx.y*BM + loadA/BK)*K + bk + loadA%BK];
        // B 是行主序 K×N:Bs[loadB/BN][loadB%BN] = B[(bk + loadB/BN)*N + blockIdx.x*BN + loadB%BN];
        // TODO(1):把上面两行写成真正的赋值语句
        As[loadA/BK][loadA%BK]=A[(blockIdx.y*BM+loadA/BK)*K+bk+loadA%BK];
        Bs[loadB/BN][loadB%BN]=B[(bk+loadB/BN)*N+blockIdx.x*BN+loadB%BN];
        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            // TODO(2) 今天的灵魂,两行:
            //   ① 从 As 取一个 a(threadRow 行、kk 列)存进普通 float
            //   ② for i in 0..TM-1: acc[i] += a * Bs[kk][threadCol + i];
            float a=As[threadRow][kk];
            for (int i=0;i<TM;i++){
                acc[i]+=a*Bs[kk][threadCol+i];
            }

        }
        __syncthreads();   // TODO(3):想想为什么这里还需要第二道屏障?删掉会怎样?
    }
    // TODO(4):把 acc[0..7] 一次性写回 C 的 globalRow 行、globalCol..+7 列
    for (int i=0;i<TM;i++){
        C[globalRow*N+globalCol+i]=acc[i];
    }
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

    dim3 block(2048);
    dim3 grid(N/BN, M/BM);

    // warmup
    matmul_regtile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(e)); return 1; }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_regtile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

    // 验证
    float cpu_c00 = 0;
    for (int k = 0; k < K; ++k) cpu_c00 += h_A[0 * K + k] * h_B[k * N + 0];
    cout << "验证 C[0][0]: GPU=" << h_C[0] << "  CPU=" << cpu_c00 << "\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}