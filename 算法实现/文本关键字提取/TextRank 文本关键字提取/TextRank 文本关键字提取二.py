"""
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
"""
import numpy as np
import jieba
import jieba.posseg as pseg


class TextRank(object):
    def __init__(self, sentence, window, alpha, max_iters):
        self.sentence = sentence
        self.window = window
        self.alpha = alpha
        self.edge_dict = dict()
        self.max_iters = max_iters

    def cut(self):
        # jieba.load_userdict('user_dict.txt')
        # tag_filter = ['a', 'd', 'n', 'v']
        allowPOS = ['a', 'd', 'n']

        seg_result = pseg.cut(self.sentence)
        self.word_list = [s.word for s in seg_result if s.flag in allowPOS]
        print(self.word_list)
        return

    def create_node(self):
        temp_list = list()
        word_list_len = len(self.word_list)
        for index, word in enumerate(self.word_list):
            if word not in self.edge_dict.keys():
                temp_list.append(word)
                temp_set = set()
                left = index - self.window + 1
                right = index + self.window
                if left < 0:
                    left = 0
                if right >= word_list_len:
                    right = word_list_len
                for i in range(left, right):
                    if i == index:
                        continue
                    temp_set.add(self.word_list[i])
                self.edge_dict[word] = temp_set
        return

    def create_matrix(self):
        self.matrix = np.zeros([len(set(self.word_list)), len(set(self.word_list))])
        self.word_index = dict()
        self.index_dict = dict()

        for i, v in enumerate(set(self.word_list)):
            self.word_index[v] = i
            self.index_dict[i] = v
        for key in self.edge_dict.keys():
            for w in self.edge_dict[key]:
                self.matrix[self.word_index[key]][self.word_index[w]] = 1
                self.matrix[self.word_index[w]][self.word_index[key]] = 1
        for j in range(self.matrix.shape[1]):
            sum = 0
            for i in range(self.matrix.shape[0]):
                sum += self.matrix[i][j]
            for i in range(self.matrix.shape[0]):
                self.matrix[i][j] /= sum
        return

    def calc_text_rank(self):
        self.text_rank = np.ones([len(set(self.word_list)), 1])
        for i in range(self.max_iters):
            self.text_rank = (1 - self.alpha) + self.alpha * np.dot(self.matrix, self.text_rank)
        return

    def print_result(self):
        text_rank = dict()
        for i in range(len(self.text_rank)):
            text_rank[self.index_dict[i]] = self.text_rank[i][0]
        ret = sorted(text_rank.items(), key=lambda x: x[1], reverse=True)
        print(ret)
        return


if __name__ == '__main__':
    s = '程序员(英文Programmer)是从事程序开发、维护的专业人员。' \
        '一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。' \
        '软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。'
    tr = TextRank(s, 3, 0.85, 700)
    tr.cut()
    tr.create_node()
    tr.create_matrix()
    tr.calc_text_rank()
    tr.print_result()
