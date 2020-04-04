"""
参考链接:
https://blog.csdn.net/koibiki/article/details/83031788

我的理解:
在矩阵的表示形式中.
字符串编辑的过程可以理解为从左上角向右下角行进.
向右表示删除一个字符使之匹配, 向下表示插入一个字符使之匹配, 向右下, 则表示修改一个字符使之匹配.
最终矩阵右下角(最后一个位置)的值就表示编辑距离.
"""
import numpy as np


def edit_distance(word1, word2):
    """动态归划实现"""
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=np.int)
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # 计算编辑距离时, len1 中的每一个字符与 len2 中的每一个字符对比, 然后取该位置的值的左边, 上边, 左上边三个值的最小值递增.
    # 如果两字符相同则递增 0, 如果不同则递增 1.
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta,
                           dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1)
    return dp[len1][len2]


def edit_distance_2(word1, word2):
    """递归实现"""
    if len(word1) == 0:
        return len(word2)
    elif len(word2) == 0:
        return len(word1)
    else:
        pass

    if word1 == word2:
        return 0

    if word1[-1] == word2[-1]:
        delta = 0
    else:
        delta = 1

    return min(edit_distance_2(word1[:-1], word2) + 1,
               edit_distance_2(word1, word2[:-1]) + 1,
               edit_distance_2(word1[:-1], word2[:-1]) + delta)


def demo1():
    ret = edit_distance('jerry', 'jary')
    print(ret)
    return


def demo2():
    ret = edit_distance_2('jerry', 'jary')
    print(ret)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
