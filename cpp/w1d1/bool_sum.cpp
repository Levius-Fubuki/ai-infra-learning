#include <iostream>
#include <math.h>
using namespace std;

bool is_prime(int n) {
    if (n < 2) return false;            // 0/1/负数 不是素数
    for (int i = 2; i * i <= n; ++i) {  // 用 i*i<=n 代替 sqrt,无浮点
        if (n % i == 0) return false;   // 用 % 取余
    }
    return true;
}

int sum(int arr[],int n){
    int s=0;
    for(int i=0;i<n;i+=1){
        s=s+arr[i];
    }
    return s;
}

int main() {
    int x;
    cin >> x;
    cout << x << " is prime? " << is_prime(x) << "\n";

    int a[] = {1, 2, 3, 4, 5};
    cout << "sum = " << sum(a, 5) << "\n";   // 应输出 15
}