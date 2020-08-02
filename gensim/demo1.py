#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://blog.csdn.net/shuihupo/article/details/85226128
"""
import multiprocessing

from gensim.models import Word2Vec, KeyedVectors, word2vec
import jieba


def gen_file_in_line(fpath, empty_line=False):
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line and not empty_line:
                continue
            yield line


def load_file(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        document = f.read().split()
    return document


def segment_file(fpath, to_fpath):
    with open(to_fpath, 'w', encoding='utf-8') as f:
        for line in gen_file_in_line(fpath):
            line_seg = ' '.join(jieba.cut(line))
            f.write(line_seg + '\n')


def demo1():
    # 生成分词后的文件.
    fpath = "../dataset/novel/人民的名义.txt"
    to_fpath = "../dataset/novel/人民的名义_segment.txt"
    segment_file(fpath, to_fpath)
    return


def demo2():
    seg_fpath = "../dataset/novel/人民的名义_segment.txt"

    sentences = word2vec.LineSentence(seg_fpath)
    model = Word2Vec(sentences, max_vocab_size=5000, size=128, window=5,
                     min_count=2, sg=1, negative=64,
                     workers=multiprocessing.cpu_count(), iter=100)
    print(f"model training done !")
    model.save("w2v_model.bin")
    model = Word2Vec.load("w2v_model.bin")
    for key in model.similar_by_word('人民', topn=10):
        print(key)

    # 模型储存与加载2
    model.wv.save("w2v_vector.bin")
    wv = KeyedVectors.load("w2v_vector.bin", mmap='r')
    for key in wv.similar_by_word('人民', topn=10):
        print(key)
    return


def demo3():
    seg_fpath = "../dataset/novel/人民的名义_segment.txt"

    sentences = word2vec.LineSentence(seg_fpath)
    model = Word2Vec(sentences, size=200, window=5, min_count=1,
                     workers=multiprocessing.cpu_count())

    print(model.vector_size)
    model.wv.n_similarity()
    # 从词索引出向量.
    print(model.wv['人民'])
    return


def demo4():
    # skip-gram
    """
    输出结果:
    ('nine', 0.9333532452583313)
    ('six', 0.8979641199111938)
    ('eight', 0.8863921165466309)
    ('two', 0.8850500583648682)
    ('seven', 0.8831043839454651)
    ('wto', 0.8829952478408813)
    ('zero', 0.8802609443664551)
    ('four', 0.8782426118850708)
    ('seattle', 0.8762422204017639)
    ('haymarket', 0.8760695457458496)
    """
    fpath = '../dataset/word2vec/dc/text8'
    document = load_file(fpath)
    print(f"load file done !")
    model = Word2Vec([document], max_vocab_size=5000, size=128, window=3,
                     min_count=3, sg=1, negative=64,
                     workers=multiprocessing.cpu_count(), iter=100)

    for key in model.similar_by_word('one', topn=10):
        print(key)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
    # demo3()
    # demo4()
