"""
https://github.com/loadingsjy/CRF
"""
import numpy as np


class CRF(object):
    def __init__(self):
        pass
        # 特征空间.
        self._feature_space = None
        # 各特征出现与各 tag 出现的权重矩阵.
        self._weight = None

    @staticmethod
    def load_data(data_path):
        """

        :param data_path:
        :return:

        demo:
        data_path = "../../dataset/part_of_speech_tagging_data/small_data/train.txt"
        sentences, tags = load_data(data_path)
        print(sentences)
        print(tags)
        """
        tags = set()
        sentence = []
        sentences = []
        word_num = 0
        words_set = set()
        with open(data_path, mode='r', encoding='utf-8') as f:
            for line in f:
                if line != '\n' and line[0] != ' ':
                    line_split = line.split()
                    word = line_split[1]
                    tag = line_split[3]
                    word_num += 1
                    tags.add(tag)
                    words_set.add(word)
                    sentence.append(tuple((word, tag)))
                else:
                    sentences.append(sentence)
                    sentence = []
        return sentences, tags

    @staticmethod
    def _bi_partial_feature_template(pre_tag):
        """

        :param pre_tag:
        :return: 如: {'01:CS'}

        demo:
        ret = bi_partial_feature_template('CS')
        print(ret)
        """
        feature_set = set()
        feature_set.add("".join(["01:", pre_tag]))
        return feature_set

    @staticmethod
    def _uni_partial_feature_template(sentence, i):
        """
        计算句子中第 i 个词的特征.
        :param sentence: 包含 (word, tag) 元素的列表. 如:
        [('戴相龙', 'NR'), ('说', 'VV'), ('中国', 'NR'), ('经济', 'NN'), ('发展', 'NN'), ('为', 'P'), ('亚洲', 'NR'), ('作出', 'VV'), ('积极', 'JJ'), ('贡献', 'NN')]
        :param i:
        :return: 如: {'06:戴相龙*说', '08:龙', '15:相龙', '14:戴相龙', '10:戴*相', '05:戴相龙*^', '07:戴', '14:戴相', '03:^^', '14:戴', '09:相', '15:戴相龙', '02:戴相龙', '11:龙*相', '04:说', '15:龙'}

        demo:
        data_path = "../../dataset/part_of_speech_tagging_data/small_data/train.txt"
        sentences, tags = load_data(data_path)
        ret = uni_partial_feature_template(sentences[0], 0)
        print(ret)
        """
        con = '_consecutive_'
        prev_ = "^^"  # 句首标志
        next_ = "$$"  # 句末标志
        word = sentence[i][0]
        word_len = len(word)
        feature_set = set()
        sentence_len = len(sentence)

        if sentence_len == 1:
            prev_word = prev_
            next_word = next_
        else:
            if i == 0:
                prev_word = prev_
                next_word = sentence[i + 1][0]
            elif i == sentence_len - 1:
                prev_word = sentence[i - 1][0]
                next_word = next_
            else:
                prev_word = sentence[i - 1][0]
                next_word = sentence[i + 1][0]

        # feature_set.add('01:' + tag + '*' + pre_tag)
        feature_set.add("".join(['02:', word]))
        feature_set.add("".join(['03:', prev_word]))
        feature_set.add("".join(['04:', next_word]))
        feature_set.add("".join(['05:', word, '*', prev_word[-1]]))
        feature_set.add("".join(['06:', word, '*', next_word[0]]))
        feature_set.add("".join(['07:', word[0]]))
        feature_set.add("".join(['08:', word[-1]]))

        for k in range(1, word_len - 1):
            feature_set.add("".join(['09:', word[k]]))
            feature_set.add("".join(['10:', word[0], '*', word[k]]))
            feature_set.add("".join(['11:', word[-1], '*', word[k]]))

        if word_len == 1:
            feature_set.add("".join(['12:', word, '*', prev_word[-1], '*', next_word[0]]))
        for k in range(word_len - 1):
            if word[k] == word[k + 1]:
                feature_set.add("".join(['13:', word[k], '*', con]))

        for k in range(1, min(5, word_len + 1)):
            feature_set.add("".join(["14:", word[0:k]]))
            feature_set.add("".join(["15:", word[-k:]]))
        return feature_set

    def _create_partial_feature_template(self, sentence, i, pre_tag):
        feature_set = self._uni_partial_feature_template(sentence, i) | self._bi_partial_feature_template(pre_tag)
        return feature_set

    @staticmethod
    def cal_score(features, weight, partial_feature_space):  # 求出该特征对于所有tag的得分
        scores = [weight[partial_feature_space[feature]] for feature in features if
                  feature in partial_feature_space]
        return np.sum(scores, axis=0)

    def create_partial_feature_space(self, sentences, tags):
        """
        遍历整个训练集, 创建特征空间.
        :param sentences:
        :param tags:
        :return:
        """
        partial_feature_space = dict()
        n = len(tags)
        BOS = '_START_'
        partial_feature_set = set()
        for sentence in sentences:
            sentence_len = len(sentence)
            for i in range(sentence_len):
                if i == 0:
                    pre_tag = BOS
                else:
                    pre_tag = sentence[i - 1][1]
                feature_set = self._create_partial_feature_template(sentence, i, pre_tag)
                partial_feature_set |= feature_set
        for index, feature in enumerate(partial_feature_set):
            partial_feature_space[feature] = index
        self._feature_space = partial_feature_space
        g = len(partial_feature_space)

        # 每一个特征的出现, 都对应地给每一个 tags 类别一个权重. 表示着特征出现, 则各类别也出现的评分.
        self._weight = np.zeros((g, n), dtype='float64')

        # bi_features = [list(bi_partial_feature_template(pre_tag)) for pre_tag in tags]
        # bi_scores = np.array([cal_score(bi_feature, weight, partial_feature_space) for bi_feature in bi_features])
        return None


if __name__ == '__main__':
    crf = CRF()
    ret = crf._bi_partial_feature_template('CS')
    print(ret)
