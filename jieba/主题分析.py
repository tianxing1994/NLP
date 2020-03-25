import glob
import os
import random

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import decomposition
from sklearn.decomposition import LatentDirichletAllocation



def load_documents():
    docs = []
    pattern = os.path.join('../dataset/others/mini_documents', "*.txt")
    for f_name in glob.glob(pattern):
        with open(f_name, 'r', encoding='gbk', errors='ignore') as f:
            words = " ".join(jieba.cut(f.read()))
            docs.append(words)
    return docs


def calc_documents_tfidf(documents):
    random.shuffle(documents)
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(documents)
    tfidf = TfidfTransformer().fit_transform(counts)

    feature_names = count_vect.get_feature_names()
    return tfidf, feature_names


def demo1():
    """
    非负矩阵分解, 作文本主题分析.
    参考链接:
    https://blog.csdn.net/jeffery0207/article/details/84348117

    输出结果:
    Topic #0:
    艺术 图书 民族 文化 舞蹈 方法 我们 书评 诗体 创作方法 云南 发展 风格 生活 现实主义 传统 形式 表现主义 诗歌 评论 文艺 内容 格律 创作 社会
    Topic #1:
    雷达 设计 双发 飞机 延程 工装 结构 定位 机场 飞行 辐射 液滴 航空公司 修理 运行 备降 元件 损伤 散射 装配工 市场 检查 蜂窝 骨架 介质
    """
    n_topic = 2
    n_top_words = 25

    docs = load_documents()
    tfidf, feature_names = calc_documents_tfidf(docs)

    nmf = decomposition.NMF(n_components=n_topic)
    nmf = nmf.fit(tfidf)

    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    return


def demo2():
    """
    LDA 主题模型
    参考链接:
    https://blog.csdn.net/u010551621/article/details/45258573
    https://blog.csdn.net/qq_40006058/article/details/85865695
    """
    n_topic = 2
    n_top_words = 25

    docs = load_documents()
    tfidf, feature_names = calc_documents_tfidf(docs)

    lda = LatentDirichletAllocation(n_components=n_topic, max_iter=200,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    lda = lda.fit(tfidf)

    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    return


if __name__ == '__main__':
    demo1()
    demo2()
