// profile_demo.cu —— 自造一个 memory-bound 和一个 compute-bound kernel
// 用 nsys 看时间线,用 ncu 看 SOL:时间到底花在访存还是计算
#include <cstdio>

// ---- kernel 1: memory-bound ----
// 每线程:读 2 个 float、写 1 个 float,只做 1 次乘加
// 12 字节访存才换 2 FLOP → 必然卡带宽
__global__ void axpy_kernel(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaf(a, x[i], y[i]);   // y = a*x + y
}

// ---- kernel 2: compute-bound ----
// 每线程:在寄存器里连环乘加 iters×4 次,几乎不碰内存
// 开 4 条独立链(x0~x3):1 条链时每个 FMA 都得等上一个结果,
// 4 条链可以流水并行,SM 才吃得饱
__global__ void compute_kernel(float* out, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.000001f, x0 = 1.f, x1 = 1.f, x2 = 1.f, x3 = 1.f;
    for (int k = 0; k < iters; ++k) {
        x0 = fmaf(x0, a, 0.001f);
        x1 = fmaf(x1, a, 0.001f);
        x2 = fmaf(x2, a, 0.001f);
        x3 = fmaf(x3, a, 0.001f);
    }
    out[i] = x0 + x1 + x2 + x3;   // 最后写一次,防止循环被编译器优化没
}

int main() {
    const int N_MEM = 1 << 26;   // 6710万 float = 256MB/数组
    const int N_CPT = 1 << 22;   // 419万线程做纯计算
    const int ITERS = 2000;
    const int block = 256;

    float *x, *y, *out;
    cudaMalloc(&x,   N_MEM * sizeof(float));
    cudaMalloc(&y,   N_MEM * sizeof(float));
    cudaMalloc(&out, N_CPT * sizeof(float));
    cudaMemset(x, 0, N_MEM * sizeof(float));
    cudaMemset(y, 0, N_MEM * sizeof(float));

    printf("[1] axpy (memory-bound): %d 元素\n", N_MEM);
    axpy_kernel<<<(N_MEM + block - 1) / block, block>>>(x, y, 2.0f, N_MEM);

    printf("[2] compute (compute-bound): %d 线程 x %d 迭代 x 4 FMA\n", N_CPT, ITERS);
    compute_kernel<<<(N_CPT + block - 1) / block, block>>>(out, ITERS);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("done\n");
    cudaFree(x); cudaFree(y); cudaFree(out);
    return 0;
}