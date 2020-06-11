#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
import jieba


def demo1():
    # 文本预处理
    to_file_path = '../dataset/novel/to_大石内藏助的一天.txt'
    file_path = '../dataset/novel/大石内藏助的一天.txt'
    pattern = re.compile("[\u4e00-\u9fa5]+")

    f_write = open(to_file_path, 'a', encoding='utf-8')
    with open(file_path, 'r', encoding='utf-8') as f_reader:
        for l in f_reader.readlines():
            line = l.replace('\r', '').replace('\n', '').strip()
            # print(line)
            if line == '' or line is None:
                continue
            line = ' '.join(jieba.cut(line))

            seg_list = pattern.findall(line)
            print(seg_list)
            f_write.write(" ".join(seg_list) + '\n')
            f_write.flush()

    f_write.close()
    return


if __name__ == '__main__':
    demo1()
