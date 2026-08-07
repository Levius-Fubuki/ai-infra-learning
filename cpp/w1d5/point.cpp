// point.cpp —— Day 5: struct/class & 模板
// 编译: g++ -std=c++17 point.cpp -o point && ./point

#include <iostream>
#include <cmath>
using namespace std;

// ---- 1. struct + 成员函数 ----
struct Point {
    double x, y;
    double dist_to(const Point& o) const {   // const:承诺不改自身
        double dx = x - o.x, dy = y - o.y;
        return sqrt(dx*dx + dy*dy);
    }
};

// ---- 2. class + 封装(默认 private)----
class Counter {
    int count = 0;                  // private:外部碰不到
public:
    void inc() { ++count; }
    int  get() const { return count; }
};

// ---- 3. 函数模板:一份代码,多类型 ----
template<typename T>
T my_add(T a, T b) { return a + b; }

// ---- 4. 类模板:Point 支持任意数值类型 ----
template<typename T>
struct PointT {
    T x, y;
    T dist_to(const PointT<T>& o) const {
        T dx = x - o.x, dy = y - o.y;
        return sqrt(dx*dx + dy*dy);
    }
};

int main() {
    cout << "===== 1. struct + 成员函数 =====" << endl;
    Point p1{0.0, 0.0}, p2{3.0, 4.0};
    cout << "p1=(" << p1.x << "," << p1.y << ") p2=(" << p2.x << "," << p2.y << ")\n";
    cout << "dist = " << p1.dist_to(p2) << "  (3-4-5 三角形 → 5)\n";

    cout << "\n===== 2. struct 连续内存(sizeof & 地址)=====" << endl;
    cout << "sizeof(Point) = " << sizeof(Point) << "  (2 个 double = 16)\n";
    cout << "&p2.x = " << &p2.x << "\n&p2.y = " << &p2.y << endl;
    cout << "相差 " << (char*)&p2.y - (char*)&p2.x << " 字节 (= sizeof(double),连续)\n";

    cout << "\n===== 3. class 封装(public/private)=====" << endl;
    Counter c;
    c.inc(); c.inc(); c.inc();
    cout << "count = " << c.get() << "  (只能通过 get() 读,不能直接 c.count)\n";

    cout << "\n===== 4. 函数模板 =====" << endl;
    cout << "my_add<int>(2,3)       = " << my_add<int>(2, 3) << endl;
    cout << "my_add<double>(1.5,2.5)= " << my_add<double>(1.5, 2.5) << endl;
    cout << "my_add(2,3)            = " << my_add(2, 3) << "  (类型可省,编译器推导)\n";

    cout << "\n===== 5. 类模板 PointT<T>(double/int 共用一份代码)=====" << endl;
    PointT<double> pd{0.0, 0.0}, pd2{3.0, 4.0};
    PointT<int>    pi{0, 0},     pi2{3, 4};
    cout << "double 点距离 = " << pd.dist_to(pd2) << endl;
    cout << "int    点距离 = " << pi.dist_to(pi2) << "  (int 版返回 int,5.0→5)\n";
    cout << "→ 编译器生成了 PointT<double> 和 PointT<int> 两份代码\n";

    cout << "\n----- 为什么 CUDA/vLLM 爱模板 -----\n";
    cout << "一份 kernel 模板同时支持 fp16/fp32/int8,编译器为每种生成特化代码,零运行时开销\n";
    return 0;
}