// vector_add.cu —— Day 6: 第一个 CUDA kernel 🎉 GPU 上做向量加法
// 编译: nvcc vector_add.cu -o vector_add && ./vector_add

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// kernel:__global__ = "GPU 上执行,从 CPU 启动",返回必须是 void
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // ⭐ 全局线程索引(必背)
    if (i < n) c[i] = a[i] + b[i];                   // 越界保护
}

int main() {
    const int N = 1 << 20;                  // 2^20 ≈ 105 万元素
    const size_t bytes = N * sizeof(float);

    // ① CPU 准备数据(Day4 的 vector,连续内存,正好整块搬)
    vector<float> h_a(N), h_b(N), h_c(N);   // h_ 前缀 = host
    for (int i = 0; i < N; ++i) { h_a[i] = i*0.001f; h_b[i] = i*0.002f; }

    // ② GPU 显存分配(cudaMalloc ≈ Day3 的 new,但在显存上)
    float *d_a, *d_b, *d_c;                  // d_ 前缀 = device
    cudaMalloc(&d_a, bytes);                 // ⚠️ 传"指针的地址" &d_a(和 new 写法不同)
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // ③ 搬数据:host → device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // ④ 启动 kernel:<<<几个 block, 每 block 几 thread>>>
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;   // 向上取整
    cout << "线程组织: " << numBlocks << " block × " << threadsPerBlock
         << " thread = " << (long)numBlocks * threadsPerBlock
         << " 个 thread(覆盖 " << N << " 元素)\n";
    add_kernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // ⑤ 搬结果:device → host(此步会自动等 kernel 算完)
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // ⑥ 释放显存(cudaFree ≈ Day3 的 delete)
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    // 验证 GPU == CPU
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            if (errors < 5) cout << "❌ i=" << i << " GPU=" << h_c[i] << " 应=" << expected << "\n";
            ++errors;
        }
    }
    cout << "N = " << N << "  "
         << (errors == 0 ? "✅ 全部正确!GPU==CPU,你点亮 GPU 了 🎉" : "❌ 有错误") << "\n";
    cout << "抽查:h_a[100]=" << h_a[100] << " + h_b[100]=" << h_b[100]
         << " = h_c[100]=" << h_c[100] << "\n";
    return 0;
}