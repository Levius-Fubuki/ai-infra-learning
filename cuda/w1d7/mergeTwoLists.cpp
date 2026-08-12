#include <iostream>
#include <vector>

using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(-1);   // ⭐ 哨兵假头(必须 new 出来!)
    ListNode* tail = dummy;               // tail 始终指向结果链末尾

    while (l1 && l2) {                    // ⭐ && 不是 ||:两个都在才比
        if (l1->val <= l2->val) {
            tail->next = l1;              // ⭐ 直接接【原节点】,不 new 新的!
            l1 = l1->next;
        } else {
            tail->next = l2;
            l2 = l2->next;
        }
        tail = tail->next;                // tail 往后挪一格
    }
    tail->next = (l1 ? l1 : l2);          // 剩下的整条接上(本身就有序)

    ListNode* head = dummy->next;         // 真链头
    delete dummy;                         // 哨兵用完归还(Day3 好习惯)
    return head;                          // ⭐ 必须 return!
}

// ---- 测试辅助 ----
ListNode* makeList(const vector<int>& v) {       // 用 vector 造链表
    ListNode* dummy = new ListNode(-1);
    ListNode* tail = dummy;
    for (int x : v) { tail->next = new ListNode(x); tail = tail->next; }
    ListNode* h = dummy->next; delete dummy; return h;
}
void printList(ListNode* h) {                     // 打印 1->2->4
    while (h) { cout << h->val; if (h->next) cout << "->"; h = h->next; }
    cout << endl;
}

int main() {
    ListNode* l1 = makeList({1, 2, 4});
    ListNode* l2 = makeList({1, 3, 4});
    printList(mergeTwoLists(l1, l2));   // 1->1->2->3->4->4
}