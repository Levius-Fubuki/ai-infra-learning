// vector_stats.cpp —— Day 4: STL 容器(vector / string / map) + 均值方差
// 编译: g++ -std=c++17 vector_stats.cpp -o vector_stats && ./vector_stats

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
using namespace std;

// const& 借用大 vector:0 拷贝(Day3 学的)
double mean(const vector<double>& v) {
    double sum = 0;
    for (double x : v) sum += x;          // 只读,小类型用值拷贝无所谓
    return sum / v.size();
}

// 样本方差(分母 n-1,数据分析标准定义)
double variance(const vector<double>& v) {
    double m = mean(v);
    double sq = 0;
    for (const double& x : v) sq += (x - m) * (x - m);   // const& 借用
    return sq / (v.size() - 1);          // 假设 size>=2
}

int main() {
    cout << "===== 1. vector ≈ Python list =====" << endl;
    vector<double> nums = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    nums.push_back(10.0);                // ≈ append
    cout << "size=" << nums.size() << "  nums[0]=" << nums[0]
         << "  back()=" << nums.back() << endl;

    cout << "\n===== 2. 均值 / 样本方差 / 标准差 =====" << endl;
    cout << "mean   = " << mean(nums) << endl;
    cout << "var    = " << variance(nums) << endl;
    cout << "stddev = " << sqrt(variance(nums)) << endl;

    cout << "\n===== 3. 范围 for:值拷贝 vs 引用(经典陷阱)=====" << endl;
    vector<double> demo = nums;          // 拷贝一份,不污染 nums
    for (double x : demo) x *= 2;        // 改副本
    cout << "for(double x)  改副本 → demo[0]=" << demo[0] << " (没变)" << endl;
    for (double& x : demo) x *= 2;       // 改原件
    cout << "for(double& x) 改原件 → demo[0]=" << demo[0] << " (翻倍了)" << endl;

    cout << "\n===== 4. 连续内存:data() 裸指针(CUDA 搬运的钥匙)=====" << endl;
    double* p = nums.data();
    cout << "nums.data()=" << p << endl;
    cout << "p[0]=" << p[0] << "  p[1]=" << p[1] << "  (连续,裸指针也能索引)" << endl;
    // Day6 预告:
    // cudaMemcpy(d_x, nums.data(), nums.size()*sizeof(double), cudaMemcpyHostToDevice);

    cout << "\n===== 5. RAII:vector 自动释放(对比 Day3 手动 delete)=====" << endl;
    {
        vector<int> tmp(1000000, 7);     // 100万个 int ≈ 4MB
        cout << "tmp 分配 " << tmp.size() << " 个 int @ " << tmp.data() << endl;
    }
    cout << "(tmp 离开花括号已自动释放,无需 delete)\n";

    cout << "\n===== 6. string(可原地改,区别于 Python str)=====" << endl;
    string s = "hello";
    s += " world";
    cout << s << "  (len=" << s.size() << ")" << endl;
    s[0] = 'H';
    cout << "改首字母 → " << s << endl;

    cout << "\n===== 7. map ≈ dict(自动按 key 排序)=====" << endl;
    map<string, int> freq;
    for (char c : s) freq[string(1, c)]++;          // 统计每个字符(含空格)
    cout << "字符频率:\n";
    for (const auto& [k, v] : freq)                  // 结构化绑定
        cout << "  '" << k << "' : " << v << endl;

    return 0;
}