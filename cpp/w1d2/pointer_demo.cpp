#include <iostream>
using namespace std;

int main() {
    // ① 变量有地址
    int x = 42;
    cout << "x 的值    = " << x  << "\n";
    cout << "x 的地址  = " << &x << "\n";   // & 取地址

    // ② 指针:存地址的变量
    int* p = &x;                            // p 指向 x
    cout << "p 的值(=x地址)= " << p  << "\n";
    cout << "*p(指向的内容) = " << *p << "\n"; // * 解引用

    // ③ 用指针间接修改
    *p = 99;                                // 顺着地址改 → x 也变了
    int y=88;
    int *py=&y;
    cout<< p << "\n";
    cout<< py <<"\n";
    cout << "执行 *p=99 后 x = " << x << "\n";   // 输出 99

    // ④ 指针遍历数组
    int arr[5] = {10, 20, 30, 40, 50};
    int* q = arr;                           // 数组名 = 首元素地址!
    cout << "数组: ";
    for (int i = 0; i < 5; ++i) {
        cout << *(q + i) << " ";            // arr[i] ≡ *(arr+i)
    }
    cout << "\n";

    // ⑤ 空指针:还没指向任何东西
    int* np = nullptr;
    cout << "np = " << np << " (空指针,不能 *np)\n";

    return 0;
}