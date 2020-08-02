#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re


def demo1():
    """
    从文本中匹配出字符串的正则表达式.
    """
    pattern = re.compile('[\u4e00-\u9fa5]+')
    string = "“什么事？”少年并未受到惊吓，昂首回答道。"
    ret = pattern.findall(string)
    print(ret)
    return


def demo2():
    pattern = re.compile('[\u4e00-\u9fa5a-zA-Z0-9]+')
    string = "不管你答不答应, I will do my BEST. to be myself NO1"
    ret = pattern.findall(string)
    print(ret)
    return


def demo3():
    pattern = re.compile('[\u4e00-\u9fa5a-zA-Z0-9]+')
    string = "不管你答不答应, I will do my BEST. to be myself NO1"
    ret = pattern.match(string)
    print(ret)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
    # demo3()
