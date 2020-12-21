### https://www.nowcoder.com/ta/job-code-high刷题网址

- [https://www.nowcoder.com/ta/job-code-high刷题网址](#https---wwwnowcodercom-ta-job-code-high----)
- [NC78：反转链表](#nc78反转链表)
- [NC93：设计LRU缓存结构](#nc93设计lru缓存结构)
- [NC105：二分查找](#nc105二分查找)
- [NC4：断链表是否有环](#nc4断链表是否有环)
- [NC3：链表环的入口节点](#nc3链表环的入口节点)
- [NC53：删除链表的倒数第n个节点](#nc53删除链表的倒数第n个节点)
- [NC45：二叉树的遍历](#nc45二叉树的遍历)
- [NC88：寻找第k大元素](#nc88寻找第k大元素)
- [NC15：二叉树的层次遍历](#nc15二叉树的层次遍历)
- [NC33： 合并两个有序链表：](#nc33-----------)
- [NC119:最小k个数](#nc119---k--)
- [NC76：用两个栈实现队列](#nc76---------)
- [NC52：括号序列](#nc52-----)
- [NC68：青蛙跳台阶](#nc68------)
- [NC127:最长公共子串](#nc127-------)
- [NC50 链表中的每k个一组翻转](#nc50------k-----)
- [NC102:在二叉树上找到两个节点的最近父节点](#nc102------------------)
- [NC22:合并两个有序数组](#nc22---------)
- [NC40:两个链表生成相加链表](#nc40-----------)
- [NC61 两数之和](#nc61-----)
- [NC41:最长无重复字符串](#nc41---------)
- [NC66 两个链表的第一个公共节点](#nc66-------------)
- [NC103:翻转字符串](#nc103------)
- [NC19:子数组的最大和](#nc19--------)
- [NC91:最长上升子序列](#nc91--------)
- [NC1大数加法](#nc1----)
- [NC32:求数的平方根](#nc32-------)
- [NC54:数组中的三个元素相加为0](#nc54------------0)
- [NC65:斐波那契数列](#nc65-------)
- [NC75:数组中只出现一次的数字](#nc75------------)
- [NC17最长回文字符串](#nc17-------)
- [NC51合并k个已经排序的链表](#nc51--k--------)
- [NC21:链表内指定区间翻转:](#nc21-----------)
- [NC24:删除有序链表中的重复元素](#nc24-------------)
- [NC48:在转动的有序数组中寻找目标值](#nc48---------------)
- [NC70:链表排序](#nc70-----)
- [NC13:二叉树的最大深度](#nc13---------)
- [NC62:平衡二叉树](#nc62------)
- [NC72:二叉树的镜像](#nc72二叉树的镜像)
- [NC14二叉树的层次遍历](#nc14叉树的层次遍历)
- [NC136:输出二叉树的右视图](#nc136二叉树的层次遍历)
- [NC121字符串排列](#nc121字符串排列)
- [NC90:设计getmin栈的操作](#nc90设计getmin栈的操作)
- [NC7:买股票的最佳时机](#nc7买股票的最佳时机)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

### NC78：反转链表

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        nex=None
        pre=None
        cur=pHead
        while cur:
#保存下一个元素
            nex=cur.next
#反转链表指针 
            cur.next=pre
#指针向后走
            pre=cur
            cur=nex
        return pre
```

### NC93：设计LRU缓存结构

题目描述

设计LRU缓存结构，该结构在构造时确定大小，假设大小为K，并有如下两个功能

set(key, value)：将记录(key, value)插入该结构

get(key)：返回key对应的value值

[要求]

\1. set和get方法的时间复杂度为O(1)

\1. 某个key的set或get操作一旦发生，认为这个key的记录成了最常使用的。

\1. 当缓存的大小超过K时，移除最不经常使用的记录，即set或get最久远的。

若opt=1，接下来两个整数x, y，表示set(x, y)

若opt=2，接下来一个整数x，表示get(x)，若x未出现过或已被移除，则返回-1

对于每个操作2，输出一个答案

示例1

输入

复制

[[1,1,1],[1,2,2],[1,3,2],[2,1],[1,4,4],[2,2]],3

返回值

复制

[1,-1]

```python
#
# lru design
# @param operators int整型二维数组 the ops
# @param k int整型 the k
# @return int整型一维数组
# def LRU(self , operators , k ):
        # write code here
import collections
import sys
class Solution:
    def __init__(self,k):
        self.dic=collections.OrderedDict()
        #根据放入数据的先后顺序排序,orderdict()传入的数据不一样，字典就不一样
        self.capacity=k#capacity()
    def set(self,key,value):#set data
        if key in self.dic:
            self.dic.pop(key)
        else:
            if self.capacity>0:
                self.capacity-=1
            else:
                #random return;delete last dic number;
                self.dic.popitem(False)
        self.dic[key]=value#store value in last dic number
    def get(self,key):#get data
        if key not in self.dic:
            return -1
        val=self.dic.pop(key)
        self.dic[key]=val#update value for dict and store last dict number
        return val
        
for line in sys.stdin.readlines():
    a= line.strip().replace(' ', '').split(']],')#分割键值对操作和容量
    k = int(a[1])#强制类型转换
    res = []#空列表
    s = Solution(k)
    for item in a[0][2:].split('],['):
        m = item.split(',')
        if m[0] == '1':#opt=1
            s.set(int(m[1]), int(m[2]))#执行set方法
        else:#否则（opt=2）执行get方法
            res.append(s.get(int(m[1])))#将返回值存入列表res中
    print(str(res).replace(' ', ''))#删除空格
```

### NC105：二分查找

题目描述

请实现有重复数字的有序数组的二分查找。

输出在数组中第一个大于等于查找值的位置，如果数组中不存在这样的数，则输出数组长度加一。

示例1：输入

5,4,[1,2,4,4,5]

返回值

3

（左右两指针，mid=left+(right-left)/2+1,right=mid,left=mid+1,return mid+1(下标加1)）

```python
class Solution {
public:
    /**
     * 二分查找
     * @param n int整型 数组长度
     * @param v int整型 查找值
     * @param a int整型vector 有序数组
     * @return int整型
     */
    int upper_bound_(int n, int v, vector<int>& a) {
        // write code here
        int left=0;
        int right=n-1;
        while(left<right){
            int mid=left+(right-left)/2;
            if(a[mid]>=v){
                if(mid==0||a[mid-1]<v){
                    //下标+1
                    return mid+1;
                }else{
                    right=mid;
                }
            }else{
                left=mid+1;
            }
        }
        return n+1;
    }
};
```

### NC4：断链表是否有环

题目描述

判断给定的链表中是否有环

扩展：

你能给出空间复杂度![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606148942604-e5a8edda-0a58-4934-906c-362843beb643.svg)的解法么？（快慢指针，快指针走两步，慢指针走一步，相等说明有环）

```python
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {#（快慢指针，快指针走两步，慢指针走一步，相等说明有环）
public:
    bool hasCycle(ListNode *head) {
        ListNode* fast=head;
        ListNode* slow=head;
        while(fast!=nullptr&&fast->next!=nullptr){
            fast=fast->next->next;
            slow=slow->next;
            if(fast==slow){
                return true;
            }
        }
        return false;
    }
};
```

### NC3：链表环的入口节点

题目描述

对于一个给定的链表，返回环的入口节点，如果没有环，返回null

拓展：

你能给出不利用额外空间的解法么？

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

#
# 
# @param head ListNode类 
# @return ListNode类
#
class Solution:
    def detectCycle(self , head ):
        # write code here
        if head is None or head.next is None or head.next.next is None:
            return None
        slow,fast=head,head
        while fast.next is not None and fast is not None:
            slow=slow.next
            fast=fast.next.next
            if slow is None or fast is None:
                return None
            if slow is fast:#meet ,circle
                slow=head
                #new slow2 ，proo of mathematical fromula
                while slow is not fast:
                    slow=slow.next
                    fast=fast.next
                return slow
        return None
                
```

### NC53：删除链表的倒数第n个节点

题目描述

给定一个链表，删除链表的倒数第n个节点并返回链表的头指针

例如，

 给出的链表为:1->2->3->4->5, n= 2.

 删除了链表的倒数第n个节点之后,链表变为1->2->3->5.

备注：

题目保证n一定是有效的

请给出请给出时间复杂度为\ O(n) *O*(*n*)的算法

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# @param head ListNode类 
# @param n int整型 
# @return ListNode类
#
class Solution:
    def removeNthFromEnd(self , head , n ):
        # fast go n ,fast and slow together go
        slow=fast=head
        for i in range(n-1):
            fast=fast.next
        if not fast.next:
            head=head.next
            return head
        fast=fast.next
        #while fast.next=null,slow location is n,delete n
        while fast.next:
            slow=slow.next
            fast=fast.next
        slow.next=slow.next.next
        return head
```

### NC45：二叉树的遍历

分别按照二叉树先序，中序和后序打印所有的节点。

示例1

输入

{1,2,3}

返回值

[[1,2,3],[2,1,3],[2,3,1]]

```python
/**
 * struct TreeNode {
 *  int val;
 *  struct TreeNode *left;
 *  struct TreeNode *right;
 * };
 */

class Solution {
public:
    /**
     * 
     * @param root TreeNode类 the root of binary tree
     * @return int整型vector<vector<>>
     */
    vector<int> pre;
    vector<int> in;
    vector<int> post;
    void preorder(TreeNode* root){
        if(root!=nullptr){
            pre.push_back(root->val);
            preorder(root->left);
            preorder(root->right);
        }
    }
     void inorder(TreeNode* root){
        if(root!=nullptr){
            inorder(root->left);
            in.push_back(root->val);
            inorder(root->right);
        }
    }
     void postorder(TreeNode* root){
        if(root!=nullptr){
            postorder(root->left);
            postorder(root->right);
            post.push_back(root->val);
        }
    }
    vector<vector<int> > threeOrders(TreeNode* root) {
        // write code here
        vector<vector<int>> res;
        preorder(root);
        res.push_back(pre);
        inorder(root);
        res.push_back(in);
        postorder(root);
        res.push_back(post);
        return res;
    }
};
```

### NC88：寻找第k大元素



有一个整数数组，请你根据快速排序的思路，找出数组中第K大的数。

给定一个整数数组a,同时给定它的大小n和要找的K(K在1到n之间)，请返回第K大的数，保证答案存在

输入元素格式：

[1,3,5,2,2],5,3

输出：

2

```python
# -*- coding:utf-8 -*-


def findKth( a, left,right, k):
        # write code here
        low=left
        high=right
        key=a[left]
        while left<right:
            while left<right and a[right]<=key:
                  right-=1
            a[left]=a[right]
            while left<right and a[left]>=key:
                left+=1
            a[right]=a[left]
        a[left]=key
        if(left==k-1):
            return a[left]
        elif(left>k-1):
            return findKth(a,low,left-1,k)
        else:
            return findKth(a,left+1,high,k)
import sys
try:
    while(True):
        line=sys.stdin.readline()
        if line=='':
            break
        #replace是变成'1','3','5','2','2','5','3'    
        #split()转化成最终的
        #先转化成['1', '3', '5', '2', '2', '5', '3']
        #        [1, 3, 5, 2, 2, 5, 3]
        lines=line.strip().replace('[','').replace(']','').split(',')
        print(lines)
        lines=list(map(int,lines))
        print(lines)
        size=lines[-2]
        k=lines[-1]
        lines = lines[:-2]
        ret=findKth(lines, 0, size-1, k)
        print(ret)
except:
    pass
```

### NC15：二叉树的层次遍历

给定一个二叉树，返回该二叉树层序遍历的结果，（从左到右，一层一层地遍历）

例如：

给定的二叉树是{3,9,20,#,#,15,7},

![image](https://cdn.nlark.com/yuque/0/2020/png/328370/1606284181628-63b6f614-1f82-46c5-8816-5510ea6f4d88.png)

该二叉树层序遍历的结果是

[

[3],

[9,20],

[15,7]

]

```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

#
# 
# @param root TreeNode类 
# @return int整型二维数组
#
class Solution:
    def levelOrder(self , root ):
        # write code here
        ans=[]
        queue=[]
        if not root: return ans
        queue.append(root)#存储根节点
        while queue:#列表不为空
            tmp=[]#临时列表
            for i in range(len(queue)):
                node=queue.pop(0)#每次将第一个元素放入 node中
                tmp.append(node.val)#左右节点添加到temp中
                if node.left:#左右节点加入到列表中
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.append(tmp)
        return ans
```

### NC33： 合并两个有序链表：

题解

https://leetcode-cn.com/problems/merge-two-sorted-lists/solution/he-bing-liang-ge-you-xu-lian-biao-by-leetcode-solu/

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
# 
# @param l1 ListNode类 
# @param l2 ListNode类 
# @return ListNode类
#
class Solution:
    def mergeTwoLists(self , l1 , l2 ):
        # write code here
        cur=ListNode(0)
        dum=cur
        while l1 and l2:
            if l1.val<=l2.val:
                cur.next=l1
                l1=l1.next
                #注意这里写成  cur.next,l1=l1,l1.next超时了，也不知道为啥，反正就分开写吧！！！！！
            else:
                cur.next=l2
                l2=l2.next
            cur=cur.next
        #cur.next=l1 if l1 else l2
        if l1==None:
            cur.next=l2
        elif l2==None:
            cur.next=l1
        return dum.next
```

### NC119:最小k个数

题目描述

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。

示例1

输入

复制

[4,5,1,6,2,7,3,8],4

返回值

复制

[1,2,3,4]

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if not tinput:
            return []
        if k > len(tinput):
            return []
        tinput = self.selectSort(tinput)
        return tinput[:k]
    
    def quickSort(self, arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        left = [item for item in arr[1:] if item <= pivot]
        right = [item for item in arr[1:] if item > pivot]
        return self.quickSort(left) + [pivot] + self.quickSort(right)
    
    def mergeSort(self, arr):
        if len(arr) <= 1: return arr
        mid = len(arr) // 2
        left = self.mergeSort(arr[:mid])
        right = self.mergeSort(arr[mid:])
        i, j = 0, 0
        res = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
        return res + left[i:] + right[j:]
        
    def insertSort(self, arr):
        if len(arr) <= 1: return arr
        for i in range(len(arr) - 1):
            temp = arr[i+1]
            j = i
            while j >= 0 and temp < arr[j]:
                arr[j+1] = arr[j]
                j -= 1
            arr[j+1] = temp
        return arr
    
    def selectSort(self, arr):
        if len(arr) <= 1:
            return arr
        for i in range(len(arr) - 1):
            min_ = i
            for j in range(i+1, len(arr)):
                if arr[j] < arr[min_]:
                    min_ = j
            arr[min_], arr[i] = arr[i], arr[min_]
        return arr
            
```

### NC76：用两个栈实现队列

题目描述

用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

思路：

栈A用来作入队列

栈B用来出队列，当栈B为空时，栈A全部出栈到栈B,栈B再出栈（即出队列）

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack1=[]
        self.stack2=[]
    def push(self, node):
        # write code here
        self.stack1.append(node)
    def pop(self):
        # return xx
        if self.stack2==[]:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
```

### NC52：括号序列

题目描述

给出一个仅包含字符'(',')','{','}','['和']',的字符串，判断给出的字符串是否是合法的括号序列

括号必须以正确的顺序关闭，"()"和"()[]{}"都是合法的括号序列，但"(]"和"([)]"不合法。

示例1

输入

复制

"["

返回值

复制

false

```python
#
# 
# @param s string字符串 
# @return bool布尔型
#
class Solution:
    def isValid(self , s ):
        # write code here
        slist=[]
        if len(s)%2==1:
            return False
        for str in s:
            if len(slist)==0 or (slist[-1]+str) not in ["()","{}","[]"]:
                slist.append(str)
            else:
                slist.pop()
        if len(slist)==0:        
            return True
```

### NC68：青蛙跳台阶

题目描述

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        if number ==1:
            return 1
        if number==2:
            return 2
        arr=[1,2]
        for i in range(2,number):
            arr.append(arr[i-1]+arr[i-2])
        return arr[number-1]
```

### NC127:最长公共子串

题目描述

给定两个字符串str1和str2,输出两个字符串的最长公共子串，如果最长公共子串为空，输出-1。

示例1

输入

复制

"1AB2345CD","12345EF"

返回值

复制

"2345"

```python
#
# longest common substring
# @param str1 string字符串 the string
# @param str2 string字符串 the string
# @return string字符串
#
class Solution:
    def LCS(self , str1 , str2 ):
        # find the max length
        if len(str1)>len(str2):
            str1,str2=str2,str1
        max_len,res=0,''
        #str2 substring are found each time frome the str1
        for i in range(len(str1)):
            if str1[i-max_len:i+1]in str2:
                #canot change,because the value of max_len 
                res=str1[i-max_len:i+1]
                max_len+=1
        if not res:
            return -1
        else:
            return res
```

### NC50 链表中的每k个一组翻转

题目描述

将给出的链表中的节点每\ k *k* 个一组翻转，返回翻转后的链表

如果链表中的节点数不是\ k *k* 的倍数，将最后剩下的节点保持原样

你不能更改节点中的值，只能更改节点本身。

要求空间复杂度 \ O(1) *O*(1)

例如：

给定的链表是1\to2\to3\to4\to51→2→3→4→5

对于 \ k = 2 *k*=2, 你应该返回 2\to 1\to 4\to 3\to 52→1→4→3→5

对于 \ k = 3 *k*=3, 你应该返回 3\to2 \to1 \to 4\to 53→2→1→4→5

输入

{1,2,3,4,5},2

返回值

{2,1,4,3,5}

https://blog.csdn.net/chenlibao0823/article/details/96558297

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# @param head ListNode类 
# @param k int整型 
# @return ListNode类
#
class Solution:
    def reverseKGroup(self , head , k ):
        # write code here
        p=head
        for _ in range(k):
            if p is None:
                return head
            p=p.next
        last=head
        p=head
        head=ListNode(0)
        for i in range(k):
            q=p
            p=p.next
            #reverse listnode
            q.next=head.next
            head.next=q
        #each timme let last point the first node
        last.next=self.reverseKGroup(p,k)
        return head.next
        
```

### NC102:在二叉树上找到两个节点的最近父节点

题目描述

给定一棵二叉树以及这棵树上的两个节点 o1 和 o2，请找到 o1 和 o2 的最近公共祖先节点。 

示例1

输入

[3,5,1,6,2,0,8,#,#,7,4],5,1

返回值

3

```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# @param root TreeNode类 
# @param o1 int整型 
# @param o2 int整型 
# @return int整型
#
class Solution:
    def lowestCommonAncestor(self , root , o1 , o2 ):
        # write code here
        if not root:
            return None
        if root.val==o1 or root.val==o2:
            return root.val
        left = self.lowestCommonAncestor(root.left,o1,o2)
        right=self.lowestCommonAncestor(root.right,o1,o2)
        if left is not None and right is not None:
            return root.val
        if left is None:
            return right
        if right is None:
            return left
        return None 
```

### NC22:合并两个有序数组

没看懂!!!

题目描述

给出两个有序的整数数组 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840530947-980473d4-778e-432b-8445-4add6c1433be.svg)和 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840530946-b804b3ae-7bfd-4a22-9a0e-8ed96f9e19ba.svg)，请将数组 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840530961-b506950f-1fe3-49bc-9ddc-d265b07d147c.svg)合并到数组 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840530967-a3d9d8da-0303-46ab-88d3-7e2780947b77.svg)中，变成一个有序的数组

注意：

可以假设 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840530963-52ea0f8a-103b-4e4b-b017-360f6700356a.svg)数组有足够的空间存放 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840530991-cf83a55e-d263-4a12-ac76-1dc9d700befc.svg)数组的元素， ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840531020-9e0a26e8-8e27-459b-9ab5-75787e04e4a4.svg)和 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840531036-3341d274-4c65-4765-b04f-3607c1e748a3.svg)中初始的元素数目分别为 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840531020-3273cff2-33e9-402e-937f-465358679456.svg)和 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1606840531020-9459dc33-59b9-4d50-b5fd-bfb8c00f8caf.svg)

```python
#
# 
# @param A int整型一维数组 
# @param B int整型一维数组 
# @return void
#
class Solution:
    def merge(self , A, m, B, n):
        # write code here
        if not A and not B: return []
        elif not A: return B
        elif not B: return A
        while m > 0 and n > 0:
            if A[m-1] >= B[n-1]:
                A[m+n-1] = A[m-1]
                m-=1
            else:
                A[m+n-1] = B[n-1]
                n-=1
        print(A)
        print(B)
        if n > 0: A[:n] = B[:n]
        return A
```

### NC40:两个链表生成相加链表

题目：

假设链表中每一个节点的值都在0~9之间，那么链表整体就可以代表一个整数。 

例如：9 -> 3 -> 7，可以代表整数937。 

给定两个这种链表的头节点head1和head2，请生成代表两个整数相加值的结果链表。 

例如：链表1为9 -> 3 -> 7，链表2为6 -> 3，最后生成新的结果链表为1 -> 0 -> 0 -> 0。

基本思路：

容易想到的方法是先将两个链表的值表示出来，然后将两个值累加起来，再根据累加结果生成一个新链表。这种方法实际是不可行的，因为链表的长度可以很长，表示的数字可以很大，容易出现int类型溢出。

方法一。利用两个栈，分别将链表1、2的值压入栈中，这样就生成了两个链表的逆序栈。将两个栈同时弹出，这样就相当于两个链表从低位到高位依次弹出，在这个过程中生成相加链表即可。注意相加过程中的进位问题。

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

#
# 
# @param head1 ListNode类 
# @param head2 ListNode类 
# @return ListNode类
#
class Solution:
    def addInList(self , head1 , head2 ):
        # write code here
        def revise(head):
            root = ListNode(-1)
            while head:
                p = head.next
                head.next = root.next
                root.next = head
                head = p
            head = root.next
            root.next = None
            del root
            return head
        head1 = revise(head1)
        head2 = revise(head2)
        head = ListNode(-1)
        add = 0
        p = head
        while head1 and head2:
            val = head1.val + head2.val + add
            add = val // 10
            cur = val % 10
            node = ListNode(cur)
            p.next = node
            p = p.next
            head1 = head1.next
            head2 = head2.next
        while head1:
            val = head1.val + add
            add = val // 10
            cur = val % 10
            node = ListNode(cur)
            p.next = node
            p = p.next
            head1 = head1.next
        while head2:
            val = head2.val + add
            add = val // 10
            cur = val % 10
            node = ListNode(cur)
            p.next = node
            p = p.next
            head2 = head2.next
        if add != 0:
            node = ListNode(add)
            p.next = node
            p = p.next
        root = head.next
        head.next = None
        del head
        head = revise(root)
        return head
 
```

### NC61 两数之和

题目描述

给出一个整数数组，请在数组中找出两个加起来等于目标值的数，

你给出的函数twoSum 需要返回这两个数字的下标（index1，index2），需要满足 index1 小于index2.。注意：下标是从1开始的

假设给出的数组中只存在唯一解

例如：

给出的数组为 {20, 70, 110, 150},目标值为90

输出 index1=1, index2=2

输入

[3,2,4],6

返回值

[2,3]

```python
# @param numbers int整型一维数组 
# @param target int整型 
# @return int整型一维数组
#
class Solution:
    def twoSum(self , numbers , target ):
        # a means: a hashmap ,list of value corresponding to the index
        a={}
        res=[]
        s=len(numbers)
        for i in range(s):
            h=target-numbers[i]
            if h in a:
                return [a[h]+1,i+1]
            a[numbers[i]]=i
        
while True:
    try:
        import sys
        ss=Solution()
        # remeber this: sys.stdin.readline().strip()
        y=str(sys.stdin.readline().strip())
        target=int(y.split(',')[-1])
        numbers=list(map(int,y.split(']')[0].split('[')[-1].split(',')))
        res=ss.twoSum(numbers, target)
        print('[%d,%d]'%(res[0],res[1]))
    except:
        break        
```

### NC41:最长无重复字符串

题目描述

给定一个数组arr，返回arr的最长无的重复子串的长度(无重复指的是所有数字都不相同)。

示例2

输入

复制

[2,2,3,4,3]

返回值

复制

3

```python
#
# 
# @param arr int整型一维数组 the array
# @return int整型
#
class Solution:
    def maxLength(self , arr ):
        # write code here
        longest_length, cur_seq = 0, []
        for i in arr:
            if i in cur_seq:
                #update longest length
                longest_length = max(longest_length, len(cur_seq))
                #start cur_seq
                start = cur_seq.index(i)
                cur_seq = cur_seq[start+1:]
            cur_seq.append(i)
        return max(longest_length, len(cur_seq))
```

### NC66 两个链表的第一个公共节点

题目描述

输入两个链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        while not pHead1 or not pHead2:
            return None
        stack1=[]
        stack2=[]
        while pHead1:
            stack1.append(pHead1)
            pHead1=pHead1.next
        while pHead2:
            stack2.append(pHead2)
            pHead2=pHead2.next
        commonNode=None
        while stack2 and stack1:
            node1=stack1.pop()
            node2=stack2.pop()
            if node1!=node2:
                break
            else:
                commonNode=node1
        return commonNode
```

### NC103:翻转字符串

```python
class Solution:
    def solve(self , str ):
        return str[::-1]
```

### NC19:子数组的最大和

题目描述

给定一个数组arr，返回子数组的最大累加和

例如，arr = [1, -2, 3, 5, -2, 6, -1]，所有子数组中，[3, 5, -2, 6]可以累加出最大的和12，所以返回12.

[要求]

时间复杂度为O(n)*O*(*n*)，空间复杂度为O(1)*O*(1)

示例1

输入

[1, -2, 3, 5, -2, 6, -1]

返回值

12

```python
#
# max sum of the subarray
# @param arr int整型一维数组 the array
# @return int整型
#
class Solution:
    def maxsumofSubarray(self , arr ):
        # num+i>0
        num=0
        for i in arr:
            if num+i>=0:
                num+=i
            else:
                num=0
        return num
```

### NC91:最长上升子序列

题目描述

给定数组arr，设长度为n，输出arr的最长递增子序列。（如果有多个答案，请输出其中字典序最小的）

示例1

输入

复制

[2,1,5,3,6,4,8,9,7]

返回值

复制

[1,3,4,8,9]

```python
#
# retrun the longest increasing subsequence
# @param arr int整型一维数组 the array
# @return int整型一维数组
#
class Solution:
    def LIS(self , arr ):
        # over time out
        if not arr:
            return 0
        if len(arr)==1:
            return 1
        max_len=0
        dp=[1 for _ in range(len(arr))]
        for i  in range(1,len(arr)):
            for j in range(0,i):                
                if arr[i]<arr[j]:
                     dp[i]=max(dp[j]+1,dp[i])
            max_len=max(dp[i],max_len)
        return max_len
```

### NC1大数加法

题目描述

以字符串的形式读入两个数字，编写一个函数计算它们的和，以字符串形式返回。

（字符串长度不大于100000，保证字符串仅由'0'~'9'这10种字符组成）

示例1

输入

复制

"1","99"

返回值

复制

"100"

说明

1+99=100 

```python
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# 计算两个数之和
# @param s string字符串 表示第一个整数
# @param t string字符串 表示第二个整数
# @return string字符串
#
class Solution:
    def solve(self , s , t ):
        # why this code note: unsupported operand type(s) for -: 'str' and 'int'
        # no different with python2.7
        i,j=len(s-1),len(t-1)
        carry=0
        res=''
        while i>=0 and j>=0:
            sum_num=(ord(s[i])-ord('0'))+(ord(t(j))-ord('0'))+carry
            res+=str(sum_num%10)
            carry=sum_num//10
            i-=1
            j-=1
        while i>=0:
            sum_num=(ord(s[i])-ord('0'))+carry
            res+=str(sum_num%10)
            carry=sum_num//10
            i-=1
        while j>=0:
            sum_num=(ord(t[j])-ord('0'))+carry
            res+=str(sum_num%10)
            carry=sum_num//10
            j-=1
        if carry>0:
            res+=str(carry)
        return res[::-1]
```

### NC32:求数的平方根

题目描述

实现函数 int sqrt(int x).

计算并返回x的平方根（向下取整）

示例1

输入

2

返回值

1

```python
#
# 
# @param x int整型 
# @return int整型
#
class Solution:
    def sqrt(self , x ):
        # write code here
        if x==1:
            return x
        left,right=1,x
        while (left<=right):
            mid=(left+right)/2
            if x/mid==mid:
                return mid
            elif x/mid<mid:
                right=mid-1
            else:
                left=mid+1
        return right
```

### NC54:数组中的三个元素相加为0

题目描述

给出一个有n个元素的数组S，S中是否有元素a,b,c满足a+b+c=0？找出数组S中所有满足条件的三元组。

注意：

\1. 三元组（a、b、c）中的元素必须按非降序排列。（即a≤b≤c）

\1. 解集中不能包含重复的三元组。

例如，给定的数组 S = {-10 0 10 20 -10 -40},解集为(-10, 0, 10) (-10, -10, 20)



```python
#
# 
# @param num int整型一维数组 
# @return int整型二维数组
#
class Solution:
    def threeSum(self , num ):
        # write code here
        n=len(num)
        if n<3:
            return []
        num.sort()
        res=[]
        for i in range(n):
            if num[i]>0:
                return res
            if i>0 and num[i]==num[i-1]:
                continue
            left=i+1
            right=n-1
            while left<right:
                if num[i]+num[left]+num[right]==0:
                    res.append([num[i],num[left],num[right]])#
                    while left<right and num[left]==num[left+1]:
                        left+=1
                    while left<right and num[right]==num[right-1]:
                        right-=1
                elif num[i]+num[left]+num[right]<0:
                    left+=1
                else:
                    right-=1
        return res           
```

### NC65:斐波那契数列

题目描述

大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。

n\leq 39*n*≤39

示例1

输入

复制

4

返回值

复制

3

```python
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n==0:
            return 0
        elif n==1:
            return 1
        else:
            num=[]
            num.append(0)
            num.append(1)
            for i in range(2,n+1):
                val=num[i-1]+num[i-2]
                num.append(val)
        return num[n]
            
```

### NC75:数组中只出现一次的数字

题目描述

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字

```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        res=[]
        for i in array:
            if array.count(i)==1:
                res.append(i)
        return res
```

### NC17最长回文字符串

题目描述

对于一个字符串，请设计一个高效算法，计算其中最长回文子串的长度。

给定字符串A以及它的长度n，请返回最长回文子串的长度。

示例1

输入

复制

"abc1234321ab",12

返回值

复制

7

```python
# -*- coding:utf-8 -*-

class Palindrome:
    def getLongestPalindrome(self, A, n):
        #each character is  taken as the center to diffuse to both sides,
        #and the symmetry position is equal in turn ,when the left and right .
#         note that there are two case of parity string
        def func(A,left,right):
            while left>=0 and right <n and A[left]==A[right]:
                left-=1
                right+=1
            return right-left-1
        res=0
        for i in range(n-1):
            res=max(res,func(A,i,i),func(A,i,i+1))
        return res
```

### NC51合并k个已经排序的链表

题目描述

合并\ k *k* 个已排序的链表并将其作为一个已排序的链表返回。分析并描述其复杂度。 

示例1

输入

复制

[{1,2,3},{4,5,6,7}]

返回值

复制

{1,2,3,4,5,6,7}

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
from heapq import heapify, heappop
# @param lists ListNode类一维数组 
# @return ListNode类
#
class Solution:
    def mergeKLists(self , lists ):
        # write code here
        #read all the node value
        h=[]
        for node in lists:
            while node:
                #attention here node.val not node
                h.append(node.val)
                node=node.next
        if not h:
            return None
        #create minest heap
        heapify(h)
        #create listnode
        root=ListNode(heappop(h))
        curnode=root
        while h:
            nextnode=ListNode(heappop(h))
            curnode.next=nextnode
            curnode=nextnode
        return root
```

### NC21:链表内指定区间翻转:

题目描述

将一个链表\ m *m* 位置到\ n *n* 位置之间的区间反转，要求时间复杂度 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1607305457677-3299b3db-5859-4396-920b-64c08e0ac008.svg)，空间复杂度 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1607305457713-54d99821-79bf-494a-aab6-67195691c771.svg)。

例如：

给出的链表为 1\to 2 \to 3 \to 4 \to 5 \to NULL1→2→3→4→5→*N**U**L**L*, ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1607305457681-b011ff02-3c2a-453f-85cb-d7e72b38e219.svg)，![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1607305457712-0237e344-9cfc-4b47-8803-13bad31c4c67.svg),

返回 1\to 4\to 3\to 2\to 5\to NULL1→4→3→2→5→*N**U**L**L*.

注意：

给出的 ![image](https://cdn.nlark.com/yuque/0/2020/svg/328370/1607305457709-40ddc9c8-d97c-4cbc-a46b-5fe44e3f34d2.svg)满足以下条件：

1 \leq m \leq n \leq 链表长度1≤*m*≤*n*≤链表长度

示例1

输入

{1,2,3,4,5},2,4

返回值

{1,4,3,2,5}

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

#
# 
# @param head ListNode类 
# @param m int整型 
# @param n int整型 
# @return ListNode类
#
class Solution:
    def reverseBetween(self , head , m , n ):
        # write code here
        if not head:
            return None
        cur,pre=head,None
#         fin the location of m
        while m>1:
            pre=cur
            cur=cur.next
            n=n-1
            m=m-1
        cur1=cur
        pre1=pre
#         reverse listnode  remember
        while n:
            thrid=cur.next
            cur.next=pre
            pre=cur
            cur=thrid
            n=n-1
        if pre1:
            pre1.next=pre
        else:
            head=pre
        cur1.next=cur
        return head
```

### NC24:删除有序链表中的重复元素

题目描述

给出一个升序排序的链表，删除链表中的所有重复出现的元素，只保留原链表中只出现一次的元素。

例如：

给出的链表为1 \to 2\to 3\to 3\to 4\to 4\to51→2→3→3→4→4→5, 返回1\to 2\to51→2→5.

给出的链表为1\to1 \to 1\to 2 \to 31→1→1→2→3, 返回2\to 32→3.

示例1

输入

{1,2,2}

返回值

{1}

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# @param head ListNode类 
# @return ListNode类
#
class Solution:
    def deleteDuplicates(self , head ):
        # write code here
        dummyHead=ListNode(0)
        dummyHead.next=head
        pre,cur=dummyHead,head
        while cur:
#             the next node exists,and repeats with the current node value
            while cur.next and cur.val == cur.next.val:
                cur=cur.next
#the last node of th previous node is the current node ,
# meaning thata the cyrrent node is  not moved and the latter node is not reoeated
            if pre.next==cur:
                pre=pre.next
            else:
# the latter node of previous node is not the current node ,meaning that the 
# current node moves and the latter node repeats
                pre.next=cur.next
            cur =cur.next
        return dummyHead.next
```

### NC48:在转动的有序数组中寻找目标值

题目描述

给出一个转动过的有序数组，你事先不知道该数组转动了多少

(例如,0 1 2 4 5 6 7可能变为4 5 6 7 0 1 2).

在数组中搜索给出的目标值，如果能在数组中找到，返回它的索引，否则返回-1。

假设数组中不存在重复项。

示例1

输入

[1],0

返回值

-1

```python
#
# 
# @param A int整型一维数组 
# @param target int整型 
# @return int整型
#
class Solution:
    def search(self , A , target ):
        # In any case, modify an ordered array, which is always half ordered
        left,right=0,len(A)-1
        while left<=right:
            mid=left+(right-left)//2
            if A[mid]==target:
                return mid
            if A[mid]>=A[left]:
                if A[left]<=target<=A[mid]:
                    right=mid-1
                else:
                    left=mid+1
            else:
                if A[mid]<=target<=A[right]:
                    left=mid+1
                else:
                    right=mid-1
        return -1
```

NC112:进制转换

```python
#
# 进制转换
# @param M int整型 给定整数
# @param N int整型 转换到的进制
# @return string字符串
#
class Solution:
    def solve(self , M , N ):
        #using string ,remember m%n and m//n
        if M==0:
            return M
        fu=False
        if M<0:
            fu=True
            M=-1*M
        s=''
#       use string
        t='0123456789ABCDEF'
        while M!=0:
            s+=t[M%N]
            M//=N
        if fu:
            s=s+'-'
        else:
            s
        return s[::-1]
```

### NC70:链表排序

题目描述

给定一个无序单链表，实现单链表的排序(按升序排序)。

示例1

输入

[1,3,2,4,5]

返回值

{1,2,3,4,5}

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

#
# 
# @param head ListNode类 the head node
# @return ListNode类
#
class Solution:
    def sortInList(self , head ):
        #storage in list
        l=[]
        p=head
        while p:
            l.append(p.val)
            p=p.next
        l.sort()
        p=head
        i=0
        while p:
            p.val=l[i]
            i+=1
            p=p.next
        return head
```

### NC13:二叉树的最大深度

题目描述

求给定二叉树的最大深度，

最大深度是指树的根结点到最远叶子结点的最长路径上结点的数量。

示例1

输入

{1,2}

返回值

2

```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# @param root TreeNode类 
# @return int整型
#
class Solution:
    def maxDepth(self , root ):
        # maxdepth and judge balance binary
        if root is None:
            return 0
        left=self.maxDepth(root.left)+1
        right=self.maxDepth(root.right)+1
        return max(right,left)
```

### NC62:平衡二叉树

题目描述

输入一棵二叉树，判断该二叉树是否是平衡二叉树。

在这里，我们只需要考虑其平衡性，不需要考虑其是不是排序二叉树

平衡二叉树（Balanced Binary Tree），具有以下性质：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。

示例1

输入

{1,2,3,4,5,6,7}

返回值

true

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # abs(left- right) <1
        if not pRoot:
            return True
        def helper(node):
#             attention here is return 0 
            if not node:
                return 0
            left=helper(node.left)
            right=helper(node.right)
            if abs(right-left)>1 or left==-1 or right==-1:
                return -1
            else:
                return max(right,left)+1
        return helper(pRoot)!=-1
层次遍历的模板为：
void bfs(TreeNode *root) {
    queue<TreeNode*> pq;
    pq.push(root);
    while (!pq.empty()) {
        int sz = pq.size();
        while (sz--) {
            TreeNode *node = pq.front(); pq.pop();
            // process node， ours tasks
            // push value to queue
            if (node->left) pq.push(node->left);
            if (node->right) pq.push(node->right);
 
        } // end inner while
    } // end outer while
}
```

### NC72:二叉树的镜像

题目描述

操作给定的二叉树，将其变换为源二叉树的镜像。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        def tra(root):
            if root ==None:
                return None
            tra(root.left)
            tra(root.right)
            root.left,root.right=root.right,root.left
        return tra(root)
```

### NC14二叉树的层次遍历



```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# @param root TreeNode类 
# @return int整型二维数组
#
class Solution:
    def zigzagLevelOrder(self , root ):
        # write code here
        if not root:
            return []
        #two size deque
        deque=[root]
        res=[]
        #record the level of tree
        flag=0
        while deque:
            #each time temp=[]
            temp=[]
            flag+=1
            for i in range(len(deque)):
                tempnode=deque.pop(0)
                temp.append(tempnode.val)
                if  tempnode.left:
                    deque.append(tempnode.left)
                if  tempnode.right:
                    deque.append(tempnode.right)
            if flag%2==0:
                temp.reverse()
            res.append(temp)
        return res
```

### NC136:输出二叉树的右视图

题目描述

请根据二叉树的前序遍历，中序遍历恢复二叉树，并打印出二叉树的右视图

示例1

输入

复制

[1,2,4,5,3],[4,2,5,1,3]

返回值

复制

[1,3,5]

```python
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# 求二叉树的右视图
# @param xianxu int整型一维数组 先序遍历
# @param zhongxu int整型一维数组 中序遍历
# @return int整型一维数组
#
class Solution:
    def solve(self , xianxu , zhongxu ):
        # write code here
        if not xianxu:
            return []
        def build(xianxu,zhongxu):
            if not xianxu or not zhongxu:
                return None
            root=TreeNode(xianxu[0])
            #if index[xianxu[0]] error:'builtin_function_or_method' object is not subscriptable
            index=zhongxu.index(xianxu[0])
            root.left=build(xianxu[1:index+1],zhongxu[:index])
            root.right=build(xianxu[index+1:],zhongxu[index+1:])
            return root
        root=build(xianxu,zhongxu)
        deque=[root]
        res=[]
        while deque:
            temp=[]
            for i in range(len(deque)):
                tempnode=deque.pop(0)
                temp.append(tempnode.val)
                if tempnode.left:
                    deque.append(tempnode.left)
                if tempnode.right:
                    deque.append(tempnode.right)
            #attention here not in for
            res.append(temp.pop())
        return res
```

### NC121字符串排列

题目描述

输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则按字典序打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

输入描述:

输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。

示例1

输入

"ab"

返回值

["ab","ba"]

```python
# -*- coding:utf-8 -*-
import itertools
class Solution:
    def Permutation(self, ss):
        # write code here
        if not ss:
            return []
        arr = sorted(set(list(map(''.join,itertools.permutations(ss)))))
        return arr
```

### NC90:设计getmin栈的操作

题目描述

实现一个特殊功能的栈，在实现栈的基本功能的基础上，再实现返回栈中最小元素的操作。

示例1

输入

[[1,3],[1,2],[1,1],[3],[2],[3]]

返回值

[1,2]

备注:

有三种操作种类，op1表示push，op2表示pop，op3表示getMin。你需要返回和op3出现次数一样多的数组，表示每次getMin的答案

1<=操作总数<=1000000

-1000000<=每个操作数<=1000000

数据保证没有不合法的操作



```python
#
# return a array which include all ans for op3
# @param op int整型二维数组 operator
# @return int整型一维数组
#
class Solution:
    def __init__(self):
        self.q=[]
        self.minv=[float('inf')]
    def getMinStack(self , ops ):
        # write code here
        if not ops:
            return  
        ans=[]
        for op in ops:
            if op[0]==1:
                self.q.append(op[1])
                self.minv.append(min(self.minv[-1],op[-1]))
            elif op[0]==2:
                self.q.pop()
                self.minv.pop()
            else:
                ans.append(self.minv[-1])
        return ans
```

### NC7:买股票的最佳时机

题目描述

假设你有一个数组，其中第\ i *i* 个元素是股票在第\ i *i* 天的价格。

你有一次买入和卖出的机会。（只有买入了股票以后才能卖出）。请你设计一个算法来计算可以获得的最大收益。

示例1

输入

复制

[1,4,2]

返回值

复制

3

```python
#
# 
# @param prices int整型一维数组 
# @return int整型
#
class Solution:
    def maxProfit(self , prices ):
        # write code 
        size=len(prices)
        if size<=1:
            return 0
        profit=0
        min_price=prices[0]
        for i in range(1,size):
            min_price=min(min_price,prices[i])
            profit=max(profit,prices[i]-min_price)
        return profit
        
```