#!/usr/bin/python3
# -*- coding: utf-8 -*-


def gen_file_in_line(fpath, encoding='utf-8', keep_blank_line=False):
    """
    :param fpath: 指向文件的路径.
    :param encoding: 编码格式.
    :param keep_blank_line: 迭代时, 是否需要输出空白行.
    :return:
    """
    with open(fpath, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line and not keep_blank_line:
                continue
            yield line


if __name__ == '__main__':
    pass
