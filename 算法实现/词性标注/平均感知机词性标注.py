"""
参考链接:
https://www.hankcs.com/nlp/averaged-perceptron-tagger.html
https://github.com/hankcs/AveragedPerceptronPython

数据集:
数据集在作者的 github 上可以下载到:
https://github.com/hankcs/AveragedPerceptronPython

注: 作者声明他此处给出的测试集太小, 所以准确度比较差.
"""
import random
from collections import defaultdict


class AveragedPerceptron(object):
    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']

    def __init__(self, max_iter=100):
        """
        self.weights: 包含字典的字典, 外层字典 key 值存储特征, value 为字典, 存储类别 (词性) 与对应权重的 key-value 对.
        self.classes: 集合, 存储训练集中提出的, 所有不同的类别 (词性).
        self._totals: 字典, 用于记录由 (feature, class) 元组组成的 key 值所对应的权重的和. 在计算平均权重时使用.
        self._timestamps: 字典, 用于记录 (feature, class) 元组组成的 key 值的权重上一次被修改的时间. 它的意义在于:
        当每一次调用 update 方法更新权重时, 都可以认为是产生了一个新的权重列表. 那么 n 次调整之后就有 n 个不同的权重列表,
        但由于每一次都只会更改一个值, 所以如果每一次都新建一个权重列表用于最后的求平均, 是非常浪费空间的.
        因此作者使用了这种方法, 为它们记录它上次改变的时间, 则在权重求和然后平均值, 只需要知道当前时间和上一次的修改的时间,
        然后做个乘法, 就实现了这一段时间里数据的求和.
        self.i: 记录的是当前时间. 需要注意的是, 在这里我们提到的时间, 时间戳其实是步数.
        """
        self._max_iter = max_iter

        self.tag_dict = dict()

        self.weights = dict()
        self.classes = set()
        self._totals = defaultdict(int)
        self._timestamps = defaultdict(int)
        self.i = 0

    def _predict(self, features):
        """
        features 中的每一个 feature 都对各类别投票. 最终选出类别得分最大的类别作为输出.
        """
        scores = defaultdict(float)
        for feature, value in features.items():
            if feature not in self.weights or value == 0:
                continue
            weights = self.weights[feature]
            for cls, weight in weights.items():
                scores[cls] += value * weight
        # 取得分最高的类别, 取最大值时, 先按得分排序, 得分一样的情况下按 cls 的字符排序.
        ret = max(self.classes, key=lambda x: (scores[x], x))
        return ret

    def update(self, truth, guess, features):
        def update_feature(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._timestamps[param]) * w
            self._timestamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            # 从 self.weights 中获取 f 的 value, 如果此键不存在, 将设置它并将其设置 value 为空字典.
            weights = self.weights.setdefault(f, dict())
            update_feature(truth, f, weights.get(truth, 0.0), 1.0)
            update_feature(guess, f, weights.get(guess, 0.0), -1.0)
        return None

    def average_weights(self):
        for feature, weights in self.weights.items():
            new_feat_weights = {}
            for cls, weight in weights.items():
                param = (feature, cls)
                total = self._totals[param]
                total += (self.i - self._timestamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[cls] = averaged
            self.weights[feature] = new_feat_weights
        return None

    @staticmethod
    def _normalize(word):
        """
        词的预处理.
        包含 '-' 且其不为第 1 个字符的, 记作 '$HYPHEN'
        所有 4 位的数字字符, 记作年份 '$YEAR'
        其它的数字记作: '$DIGITS'
        所有的词转换为小写.
        :param word:
        :return:
        """
        if '-' in word and word[0] != '-':
            return '$HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '$YEAR'
        elif word[0].isdigit():
            return '$DIGITS'
        else:
            return word.lower()

    @staticmethod
    def _make_tag_dict(sentences):
        """
        制作词性标记的字典.
        因为大部分词的词性都是固定的, 所以将这些词性固定的词记录下来, 直接查字典进行词性标注就行.
        而对于一些具有多词性的词, 则是需要通过模型训练来识别的.
        """
        classes = set()
        tag_dict = dict()
        counts = defaultdict(lambda: defaultdict(int))
        for words, tags in sentences:
            for word, tag in zip(words, tags):
                counts[word][tag] += 1
                classes.add(tag)

        freq_threshold = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # 出现了足够多次, 且词性固定的词, 则被认为是 "具有固定词性的词".
            if n >= freq_threshold and (float(mode) / n) >= ambiguity_thresh:
                tag_dict[word] = tag
        return classes, tag_dict

    def _get_features(self, i, word, context, prev, prev2):
        """
        根据词, 及其上下文, 等信息, 生成词特征.
        :param i: 当前词在句子中的索引.
        :param word: 当前词.
        :param context: 当前词所在句子的内容.
        :param prev: 当前词的前一位词的词性.
        :param prev2: 当前词的前两位词的词性.
        :return:
        """
        def add(name, *args):
            """将特征制作成字符串. """
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')

        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i - 1])
        add('i-1 suffix', context[i - 1][-3:])
        add('i-2 word', context[i - 2])
        add('i+1 word', context[i + 1])
        add('i+1 suffix', context[i + 1][-3:])
        add('i+2 word', context[i + 2])
        return features

    def evaluation(self, words, tags):
        """计算采用模型预测的词中, 预测正确的数量, 和总的数量. """
        correct = 0.
        total = 0.
        prev, prev2 = self.START
        context = self.START + [self._normalize(w) for w in words] + self.END
        for i, word in enumerate(words):
            tag = self.tag_dict.get(word)
            if not tag:
                feats = self._get_features(i, word, context, prev, prev2)
                tag = self._predict(feats)
            if tag == tags[i]:
                correct += 1.
            total += 1.
            prev2 = prev
            prev = tag
        return correct, total

    def predict(self, words):
        """预测一个句子的标记. """
        tokens = list()
        prev, prev2 = self.START
        context = self.START + [self._normalize(w) for w in words] + self.END
        for i, word in enumerate(words):
            tag = self.tag_dict.get(word)
            if not tag:
                feats = self._get_features(i, word, context, prev, prev2)
                tag = self._predict(feats)
            tokens.append((word, tag))
            prev2 = prev
            prev = tag
        return tokens

    def fit(self, sentences):
        classes, tag_dict = self._make_tag_dict(sentences)
        self.classes = classes
        self.tag_dict = tag_dict

        for iter_ in range(self._max_iter):
            print(f"training, iter: {iter_}")
            for words, tags in sentences:
                prev, prev2 = self.START
                context = self.START + [self._normalize(w) for w in words] + self.END
                for i, word in enumerate(words):
                    guess = self.tag_dict.get(word)
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess = self._predict(feats)
                        self.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
            random.shuffle(sentences)
        self.average_weights()
        return None

    def tag(self, corpus):
        s_split = lambda t: t.split('\n')
        w_split = lambda s: s.split()

        def split_sents(corpus):
            for s in s_split(corpus):
                yield w_split(s)

        prev, prev2 = self.START
        tokens = []
        for words in split_sents(corpus):
            context = self.START + [self._normalize(w) for w in words] + self.END
            for i, word in enumerate(words):
                tag = self.tag_dict.get(word)
                if not tag:
                    features = self._get_features(i, word, context, prev, prev2)
                    tag = self._predict(features)
                tokens.append((word, tag))
                prev2 = prev
                prev = tag
        return tokens


def load_data(data_path):
    sentences = list()
    sentence = ([], [])
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            params = line.split()
            if len(params) != 2:
                continue
            sentence[0].append(params[0])
            sentence[1].append(params[1])

            if params[0] == '.':
                sentences.append(sentence)
                sentence = ([], [])
    return sentences


def demo1():
    # 训练并在测试集上查看效果.
    train_data_path = '../../dataset/part_of_speech_tagging_data/others/train.txt'
    test_data_path = '../../dataset/part_of_speech_tagging_data/others/test.txt'

    sentences = load_data(train_data_path)
    print(f"load train data done: {sentences}")

    ap = AveragedPerceptron(max_iter=100)
    ap.fit(sentences)

    test_sentences = load_data(test_data_path)
    print(f"load test data done: {sentences}")
    for words, tags in test_sentences:
        ret = ap.evaluation(words, tags)
        print(ret)
        # break
    return


def demo2():
    # 训练的模型作英文句子的词性标注
    train_data_path = '../../dataset/part_of_speech_tagging_data/train.txt'
    test_data_path = '../../dataset/part_of_speech_tagging_data/test.txt'

    sentences = load_data(train_data_path)
    print(f"load train data done: {sentences}")

    ap = AveragedPerceptron(max_iter=100)
    ap.fit(sentences)

    test_sentences = 'an initial evaluation of the programme'
    ret = ap.tag(test_sentences)
    print(ret)
    return


if __name__ == '__main__':
    demo1()
