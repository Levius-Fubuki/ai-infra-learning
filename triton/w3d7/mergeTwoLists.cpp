#include <iostream>
#include <vector>
using namespace std;

void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int i=m-1, j=n-1, k=m+n-1;
    while(j>=0){ if(i>=0 && nums1[i]>nums2[j]) nums1[k--]=nums1[i--];
                else                          nums1[k--]=nums2[j--]; }
}

int main(){
    vector<int> nums1={1,2,4,0,0,0};
    vector<int> nums2={3,5,6};
    int m=3,n=3;
    merge(nums1,m,nums2,n);
    for (int i=0;i<m+n;i++){
        cout<<nums1[i]<<" ";
    }
    cout<<endl;
}