#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

vector<int> twoSum(const vector<int>& nums, int target) {
    unordered_map<int, int> seen;                 // 值 → 下标
    for (int i = 0; i < (int)nums.size(); ++i) {  // ← (int) 见"要改3"
        int need = target - nums[i];
        auto it = seen.find(need);                // O(1) 查找
        if (it != seen.end())                     // 找到
            return {it->second, i};               // it->second = 之前那个下标
        seen[nums[i]] = i;                        // 没找到,记下 值→下标
    }
    return {};
}

int main() {
    vector<int> nums = {2, 7, 11, 15};
    auto ans = twoSum(nums, 9);
    cout << "(" << ans[0] << "," << ans[1] << ")\n";   // (0,1)
}