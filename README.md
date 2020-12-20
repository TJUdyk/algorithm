# algorithm
so ,we all konw algorithm is very import for Algorithm engineer.
what we want to do one thing is to do at least one algorithm everyday.
215. 数组中的第K个最大元素
难度中等828收藏分享切换为英文接收动态反馈
在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
示例 1:
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
from typing import List
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
    # 使用容量为 k 的小顶堆
    # 元素个数小于 k 的时候，放进去就是了
    # 元素个数大于 k 的时候，小于等于堆顶元素，就扔掉，大于堆顶元素，就替换
        size=len(nums)
        if k>size:
            raise Exception("wrrong")
        L=[]
        for index in range(k):
            #heapq 默认最小堆
            heapq.heappush(L, nums[index])
        for index in range(k,size):
            top=L[0]
            if nums[index]>top:
                # 看一看堆顶的元素，只要比堆顶元素大，就替换堆顶元素
                heapq.heapreplace(L, nums[index])
        # 最后堆顶中的元素就是堆中最小的，整个数组中的第 k 大元素
        return L[0]
