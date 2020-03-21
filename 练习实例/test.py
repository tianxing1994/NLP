import os
from collections import defaultdict

import codecs
import jieba
import numpy as np


def is_chinese(str):
    for ch in str:
        if not (u'\u4e00' <= ch <= u'\u9fff'):
            return False
        else:
            return True


def length_requirements(str, min_length=2):
    if len(str) < min_length:
        return False
    else:
        return True


def organize_document(document):
    """
    仅保留中文词
    :return:
    """
    ret = list()
    for word in document:
        if is_chinese(word) and length_requirements(word):
            ret.append(word)
    return ret


def load_documents():
    documents = list()
    labels = list()
    file_path = '../dataset/others/mini_documents'
    names = os.listdir(file_path)
    for name in names:
        label, _ = name.split(sep='-')
        filename = os.path.join(file_path, name)
        with codecs.open(filename, 'r', encoding='gbk', errors='ignore') as f:
            content = f.read()

            document = jieba.lcut(content, cut_all=True)
            document = organize_document(document)

            documents.append(document)
            labels.append(label)
    return documents, labels


def feature_select(data):
    """
    此处将每个文档单独处理计算 TF-IDF.
    在文档分类时, 应考虑将同一类文档当作一个文档来处理.
    :param data:
    :return:
    """
    # 统计整个数据集中各词的数量.
    total_words_count = defaultdict(int)
    for doc in data:
        for word in doc:
            total_words_count[word] += 1

    # 计算整个数据集中各词的频率 TF.
    words_tf = dict()
    for word in total_words_count.keys():
        words_tf[word] = total_words_count[word] / sum(total_words_count.values())

    # 计算整个数据集中各词的逆向词频 IDF.
    doc_num = len(data)
    # 存储各词的 IDF 值.
    words_idf = dict()
    # 存储包含各词的文档数.
    words_doc = defaultdict(int)
    for word in total_words_count.keys():
        for sample in data:
            if word in sample:
                words_doc[word] += 1

    for word in total_words_count.keys():
        words_idf[word] = np.log(doc_num / (words_doc[word] + 1))

    words_tfidf = dict()
    for word in total_words_count.keys():
        words_tfidf[word] = words_tf[word] * words_idf[word]

    # 对字典按值由大到小排序.
    dict_feature_select = sorted(words_idf.items(), key=lambda x: x[1], reverse=True)
    return dict_feature_select


def demo1():
    data, target = load_documents()
    features = feature_select(data)

    print(features)
    print(len(features))
    return


if __name__ == '__main__':
    demo1()
