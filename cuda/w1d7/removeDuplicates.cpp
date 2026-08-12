#include <iostream>
#include <vector>

using namespace std;

int removeDuplicates(vector<int>& nums){
    int n=nums.size();
    if (n<=1){
        return n;
    }
    int l=1;
    for(int r=1;r<n;r++){
        if (nums[r]!=nums[r-1]){
            nums[l]=nums[r];
            l++;
        }
    }
    for(int i=0;i<n;i++){
        cout<<nums[i]<<" ";
    }
    cout<<endl;
    return l;
}

int main(){
    vector<int> nums={0,0,1,1,2};
    int ans=removeDuplicates(nums);
    cout<<ans<<endl;
}