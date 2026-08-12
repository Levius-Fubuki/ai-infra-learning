// print_ids.cu —— Week2 Day1: 打印每个线程的身份
// 编译: nvcc print_ids.cu -o print_ids && ./print_ids

#include <iostream>
#include <cstdio>
using namespace std;

__global__ void print_ids() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("[block %d / thread %d]  → global i = %d\n",
           blockIdx.x, threadIdx.x, i);
}

int main() {
    cout << "启动 4 block × 8 thread = 32 个线程\n";
    cout << "(blockDim.x=8, gridDim.x=4)\n";
    cout << "----------------------------------\n";
    print_ids<<<4, 10>>>();          // 4 block, 每 block 8 thread
    cudaDeviceSynchronize();        // ⭐ 关键!见下面解释
    return 0;
}