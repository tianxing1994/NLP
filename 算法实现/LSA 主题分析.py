#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://www.jianshu.com/p/a3d78abcff51
"""
import os
import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def re_filter(string):
    # pattern = re.compile('[\u4e00-\u9fa5a-zA-Z0-9]+')
    pattern = re.compile('[\u4e00-\u9fa5]+')

    m = pattern.findall(string)
    ret = ' '.join(m)
    return ret


def load_document(filename):
    with open(filename, 'r', encoding='gbk', errors='ignore') as f:
        content = f.read()
        document = jieba.lcut(content, cut_all=False)
    return document


def load_documents(file_path):
    names = os.listdir(file_path)
    documents = list()
    for name in names:
        label, _ = name.split(sep='-')
        filename = os.path.join(file_path, name)
        document = load_document(filename)
        document = list(map(re_filter, document))
        document = ' '.join([x for x in document if x != ''])
        documents.append(document)
    return documents


def extract_features(documents, max_features=20):
    vectorizer = CountVectorizer(max_features=max_features)
    vocabulary_matrix = vectorizer.fit_transform(documents)
    print(vectorizer.vocabulary_)

    tf_idf_transformer = TfidfTransformer(norm=None)
    tf_idf = tf_idf_transformer.fit_transform(vocabulary_matrix)
    ret = tf_idf.toarray()
    return ret


def svd(a, top_k):
    u, o, v = np.linalg.svd(a)
    print(u.shape)
    print(o.shape)

    print(v.shape)

    u_ = u[:, :top_k]
    o_ = np.diag(o[:top_k])
    v_ = v[:top_k, :]
    return u_, o_, v_


def demo1():
    file_path = '../dataset/others/mini_documents'
    documents = load_documents(file_path)
    a = extract_features(documents, max_features=4)
    u_, o_, v_ = svd(a, top_k=2)
    print(u_)
    print(o_)
    print(v_)
    return


if __name__ == '__main__':
    demo1()
