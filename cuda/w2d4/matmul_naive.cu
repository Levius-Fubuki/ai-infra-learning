#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

__global__ void matmul_naive(const float* A,const float* B,float* C,
                            int M,int N,int K){
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    int col=blockDim.y*blockIdx.y+threadIdx.y;
    if(row<M && col<N){
        float sum=0.0f;
        for(int k=0;k<K;k++){
            sum+=A[row*K+k]*B[k*N+col];
        }
        C[row*N+col]=sum;
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
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y); //64*64

    // warmup(第一次跑有初始化开销,不计时)
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 正式计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms / 1000.0) / 1e12;

    cout << "naive matmul " << M << "x" << N << "x" << K << "\n";
    cout << "耗时 = " << ms << " ms\n";
    cout << "算力 = " << tflops << " TFLOPS\n";
    cout << "3090 FP32 峰值 ~35 TFLOPS → 利用率 " << tflops / 35.0 * 100 << " %\n";

    // 验证 C[0][0]
    float cpu_c00 = 0;
    for (int k = 0; k < K; ++k) cpu_c00 += h_A[0 * K + k] * h_B[k * N + 0];
    cout << "验证 C[0][0]: GPU=" << h_C[0] << "  CPU=" << cpu_c00 << "\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}