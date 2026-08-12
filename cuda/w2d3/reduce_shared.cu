#include <iostream>
#include <vector>
using namespace std;

__global__ void reduce_shared(const float* x,float* sum,int n){
    __shared__ float s[256];

    int tid=threadIdx.x;
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    s[tid]=(i<n) ? x[i] : 0.0f;
    __syncthreads();

    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            s[tid]+=s[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        atomicAdd(sum,s[tid]);
    }
}

int main() {
    const int N = 1 << 20;
    vector<float> h_x(N, 1.0f);

    float *d_x, *d_sum;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_shared<<<numBlocks, threadsPerBlock>>>(d_x, d_sum, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float h_sum = 0;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "shared memory + 归约树:\n";
    cout << "  和 = " << h_sum << "  (正确值 " << (float)N << ")\n";
    cout << "  耗时 = " << ms << " ms\n";
    cout << "  (atomic 版是 18 ms,对比一下)\n";

    cudaFree(d_x); cudaFree(d_sum);
    return 0;
}