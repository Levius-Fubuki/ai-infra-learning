#include <iostream>
#include <vector>
using namespace std;

int remove(vector<int>& nums,int val){
    int slow=0;
    for(int fast=0; fast<nums.size(); ++fast)
        if (nums[fast]!=val) nums[slow++]=nums[fast];
    return slow;
}

int main(){
    int val=2;
    vector<int> nums={0,1,2,2,3,1,2,4};
    int ans=remove(nums,val);
    cout<<ans<<endl;
    for(int i=0;i<ans;i++){
        cout<<nums[i]<<" ";
    }
    cout<<endl;
}