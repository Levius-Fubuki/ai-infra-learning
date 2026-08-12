#include <iostream>
#include <vector>
#include <ctime>
using namespace std;

__global__ void reduce_atomic(const float* x,float* sum,int n){
    int i=blockIdx.x * blockDim.x+threadIdx.x;
    if (i<n){
        atomicAdd(sum,x[i]);
    }
}

int main() {
    const int N = 1 << 20;   // 100 万个数
    vector<float> h_x(N);
    for (int i = 0; i < N; ++i) h_x[i] = 1.0f;   // 全填 1,正确和应该是 N

    float *d_x, *d_sum;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));          // sum 初始为 0

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 计时(用 CUDA 事件,精确测 GPU 时间)
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_atomic<<<numBlocks, threadsPerBlock>>>(d_x, d_sum, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float h_sum = 0;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "atomicAdd 版本:\n";
    cout << "  和 = " << h_sum << "  (正确值 " << (float)N << ")\n";
    cout << "  耗时 = " << ms << " ms\n";

    cudaFree(d_x); cudaFree(d_sum);
    return 0;
}