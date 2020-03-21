import os
import re

import jieba


def split(str):
    pattern = re.compile('(.+) ([\d\.]+)')
    match = re.match(pattern, str)

    word = match.group(1)
    score = match.group(2)
    return word, float(score)


def load_tfidf():
    dictionary = dict()
    tfidf_path = "../../../dataset/nlp_data/tf-idf.txt"
    with open(tfidf_path, 'r', encoding='utf-8', errors='ignore') as f:
        word_list = f.readlines()
        # i = 0
        for word_score in word_list:
            # print(word)
            word, score = split(word_score)
            # print(word, score)
            dictionary[word] = score
            # print(i)
            # i += 1
    return dictionary


def get_keywords(filename, dictionary, top_k=10):
    with open(filename, 'r', encoding='gbk', errors='ignore') as f:
        content = f.read()
        document = jieba.lcut(content, cut_all=True)
        words = set(document)
        keyword = list()
        for word in words:
            if word in dictionary:
                score = dictionary[word]
                keyword.append((word, score))
        keywords = sorted(keyword, key=lambda x: x[1], reverse=True)
    return keywords[:top_k]


def demo1():
    dictionary = load_tfidf()

    file_path = '../../../dataset/others/mini_documents'
    names = os.listdir(file_path)
    for name in names:
        label, _ = name.split(sep='-')
        filename = os.path.join(file_path, name)
        keywords = get_keywords(filename, dictionary)
        print(name)
        print(keywords)
    return


if __name__ == '__main__':
    demo1()

