#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
http://norvig.com/spell-correct.html
"""
import re
from collections import Counter


def words(text):
    return re.findall(r'\w+', text.lower())


WORDS = Counter(words(open('../dataset/english_document/big.txt').read()))


def P(word, N=sum(WORDS.values())):
    """每个词出现的概率. """
    return WORDS[word] / N


def correction(word):
    """概率最大的拼写单词. """
    return max(candidates(word), key=P)


def candidates(word):
    """
    known([word]): 单词是否在 WORDS 中,
    known(edits1(word)): 单词的一次编辑是否在 WORDS 中,
    known(edits2(word)): 单词的两次编辑是否在 WORDS 中,
    """
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    """返回 words 中同时在 WORDS 中的单词. """
    return set(w for w in words if w in WORDS)


def edits1(word):
    """计算单词一次编辑(删除, 交换位置, 替换, 插入)的可能结果. """
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """计算单词两次编辑的可能结果. """
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


if __name__ == '__main__':
    ret = correction('amzon')
    print(ret)
