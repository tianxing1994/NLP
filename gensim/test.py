#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://blog.csdn.net/shuihupo/article/details/85226128
"""
import multiprocessing

from gensim.models import Word2Vec, KeyedVectors, word2vec
import jieba


def load_file(fpath, empty_line=False):
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line and not empty_line:
                continue
            yield line


def segment_file(fpath, to_fpath):
    with open(to_fpath, 'w', encoding='utf-8') as f:
        for line in load_file(fpath):
            line_seg = ' '.join(jieba.cut(line))
            f.write(line_seg + '\n')


def demo1():
    # 生成分词后的文件.
    fpath = "../dataset/novel/人民的名义.txt"
    to_fpath = "../dataset/novel/人民的名义_segment.txt"
    segment_file(fpath, to_fpath)
    return


def demo2():
    seg_fpath = "../dataset/novel/织田信长_segment.txt"

    sentences = word2vec.LineSentence(seg_fpath)
    model = Word2Vec(sentences, size=2, window=5, min_count=1,
                     workers=multiprocessing.cpu_count(), sg=1)

    model.save("w2v_model.bin")
    model = Word2Vec.load("w2v_model.bin")
    for key in model.similar_by_word('信长', topn=10):
        print(key)

    # 模型储存与加载2
    model.wv.save("w2v_vector.bin")
    wv = KeyedVectors.load("w2v_vector.bin", mmap='r')
    # for key in wv.similar_by_word('织田信长', topn=10):
    #     print(key)

    return


if __name__ == '__main__':
    # demo1()
    demo2()
