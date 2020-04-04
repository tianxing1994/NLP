"""
TextRank 文本关键字提取二.py 中的实现是抄了参考链接中的. 此处对其编码进行了调整.

参考链接:
https://blog.csdn.net/y12345678904/article/details/77855936

TextRank 的主要思想如下, 但具体的公式中有一个阻尼系数, 我并不十分明白它的用意.
算法原理:
1. 将文档分词, 形成词的列表. 只保留形容词, 副词, 名词这些有意义的词.
2. 用窗口大小 window 对列表进行遍历. 创建字典, key 值是当前词, value 是窗口中出现的其它词的集合
(由于某个词在列表中不止出现一次, 所以 value 集合中的词会在遍历过程中不断增加).
3. 以上形成的字典, 其实是表达每一个词是一个节点. 而该节点与其它的词 (节点) 相连接. 它们形成一张网.
4. 网中的每个节点起始都有 1 的分数, 现在开始迭代, 它们将自己的这 1 分平均地投给其相邻节点.
5. 4 中的过程不断迭代, 最终拥有相邻节点越多的节点将具有更高的得分.

我的评价:
我感觉, 这种算法其实就是在看哪些词出现的次数比较多. 出现次数多当然就更容易有更多的邻居词,
当然那些总是和相同的词一起出现的词, 虽然出现的次数多, 但权重依然会比较低.
"""
from collections import defaultdict
import numpy as np
import jieba.posseg as pseg


class TextRank(object):
    def __init__(self, window_size=3, damping=0.85, max_iters=700):
        # damping: 没有阻尼系数, 结果会一直震荡, 不会收敛.
        self._window_size = window_size
        self._damping = damping
        self._max_iters = max_iters

        self._word_list = None
        self._unique_words = None
        self._unique_words_num = None

        self._graph = defaultdict(set)
        self._matrix = None

    def textrank(self, sentence, allow_pos=None):
        """
        对中文句子分词, 并计算每个词的权重, 返回权重列表与对应的词列表.
        """
        self._init_word_list(sentence, allow_pos=allow_pos)
        self._init_node()
        self._init_matrix()
        text_rank = self._calc_text_rank()
        self.print_result(text_rank)
        return text_rank, self._unique_words

    def _init_word_list(self, sentence, allow_pos=None):
        """
        由于要使用词性筛选有意义的词, 在 jieba.posseg.cut 方法中不能像 jieba.cut 一样指定有并模式分词.
        """
        if allow_pos is None:
            allow_pos = ('a', 'd', 'n')
        seg_result = pseg.cut(sentence)
        self._word_list = [s.word for s in seg_result if s.flag in allow_pos]
        print(self._word_list)
        return

    def _init_node(self):
        word_list_len = len(self._word_list)

        for i, word in enumerate(self._word_list):
            left = max(i - self._window_size + 1, 0)
            right = min(i + self._window_size, word_list_len)
            neighbor = self._word_list[left: right]
            self._graph[word].update(neighbor)
        return

    def _init_matrix(self):
        self._unique_words = list(self._graph.keys())
        self._unique_words_num = len(self._unique_words)
        self._matrix = np.zeros(shape=(self._unique_words_num, self._unique_words_num), dtype=np.float64)
        for i, key_word in enumerate(self._unique_words):
            for value_word in self._graph[key_word]:
                j = self._unique_words.index(value_word)
                self._matrix[i, j] = 1
                self._matrix[j, i] = 1
        for k in range(self._unique_words_num):
            self._matrix[:, k] /= np.sum(self._matrix[:, k])
        return

    def _calc_text_rank(self):
        text_rank = np.ones(shape=(self._unique_words_num, 1), dtype=np.float64)
        for i in range(self._max_iters):
            text_rank = (1 - self._damping) + self._damping * np.dot(self._matrix, text_rank)
        return text_rank

    def print_result(self, text_rank):
        text_rank_dict = dict()
        for i in range(self._unique_words_num):
            text_rank_dict[self._unique_words[i]] = text_rank[i][0]
        ret = sorted(text_rank_dict.items(), key=lambda x: x[1], reverse=True)
        print(ret)
        return


if __name__ == '__main__':
    sentence = '程序员(英文Programmer)是从事程序开发、维护的专业人员。' \
        '一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。' \
        '软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。'
    tr = TextRank()
    tr.textrank(sentence=sentence)
