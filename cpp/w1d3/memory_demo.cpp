// memory_demo.cpp  —— Day 3: 引用 & 栈/堆内存模型
// 编译: g++ -g memory_demo.cpp -o memory_demo && ./memory_demo
// (加 -g 是为了 valgrind 能指到行号)

#include <iostream>
using namespace std;

// ---- 三种传参:值 / 指针 / 引用 ----
void addOne_value(int n)  { n = n + 1; }    // 值传递:改副本,外部不变
void addOne_ptr  (int* p) { *p = *p + 1; }  // 指针:要写 * 和 &
void addOne_ref  (int& n) { n = n + 1; }    // 引用:语法像值,效果是引用

// ---- swap 对比:引用版(最常用,语法干净)----
void swap_ref(int& a, int& b) { int t = a; a = b; b = t; }

int main() {
    cout << "===== 1. 引用是别名(地址相同)=====" << endl;
    int x = 10;
    int& r = x;          // r 是 x 的别名
    r = 20;
    cout << "x=" << x << " r=" << r << "  (&x=" << &x << " &r=" << &r << " 应相同)\n";

    cout << "\n===== 2. 引用不可换绑(b=c 是赋值,不是改绑)=====" << endl;
    int a = 1, c = 100;
    int& b = a;          // b 绑定 a
    cout << "绑定后 : a=" << a << " b=" << b << " c=" << c << endl;
    b = c;               // ⚠️ 把 c 的值赋给 a,不是改绑!
    cout << "b=c 后 : a=" << a << " b=" << b << " c=" << c << "  (a 被改成 100!)\n";

    cout << "\n===== 3. 值/指针/引用 三种传参对比 =====" << endl;
    int v = 5;
    addOne_value(v); cout << "value → v=" << v << "  (不变)\n";
    addOne_ptr (&v); cout << "ptr   → v=" << v << "  (调用要写 &v)\n";
    addOne_ref (v);  cout << "ref   → v=" << v << "  (调用像值传)\n";

    cout << "\n===== 4. swap 引用版 =====" << endl;
    int m = 1, n = 2;
    swap_ref(m, n);
    cout << "m=" << m << " n=" << n << "  (应互换)\n";

    cout << "\n===== 5. 栈 vs 堆(看地址范围差异)=====" << endl;
    int stack_var = 42;              // 栈上
    int* heap_ptr = new int(42);     // 堆上
    cout << "栈变量: 地址 " << &stack_var << "  值 " << stack_var << endl;
    cout << "堆对象: 地址 " << heap_ptr   << "  值 " << *heap_ptr  << endl;
    cout << "(观察:栈地址和堆地址的数值范围通常差很远)\n";
    delete heap_ptr;                 // 归还堆内存
    heap_ptr = nullptr;              // 防悬垂

    cout << "\n===== 6. new[] / delete[] 堆数组 =====" << endl;
    int k = 5;
    int* arr = new int[k];
    for (int i = 0; i < k; ++i) arr[i] = i * i;     // 0 1 4 9 16
    cout << "堆数组: ";
    for (int i = 0; i < k; ++i) cout << arr[i] << " ";
    cout << "\n";
    delete[] arr;                    // ⚠️ 数组用 delete[]
    arr = nullptr;

    cout << "\n===== 7. 故意泄漏(演示用!别在真实代码写)=====" << endl;
    int* leaked = new int(999);      // 没有 delete → 函数结束指针丢失 → 永久泄漏
    cout << "泄漏了 4 字节 @ " << leaked << ",本函数结束后再也 delete 不了\n";
    // (真实代码这里要写: delete leaked;)

    cout << "\n----- Day 6 预告:C++ 堆 ↔ CUDA 显存,心智模型相同 -----\n";
    cout << "C++ 堆   : new        / delete\n";
    cout << "CUDA 显存: cudaMalloc / cudaFree  ← 同款操作,只是内存在 GPU 那边\n";
    return 0;
}