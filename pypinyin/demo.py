# -*- coding: utf-8 -*-
from pypinyin import pinyin, lazy_pinyin, Style


def demo1():
    """
    :return: [['zhōng'], ['xīn']]
    """
    ret = pinyin('中心')
    print(ret)
    return


def demo2():
    """
    :return: [['zhōng', 'zhòng'], ['xīn']]
    """
    ret = pinyin('中心', heteronym=True)
    print(ret)
    return


def demo3():
    """
    :return: [['z'], ['x']]
    """
    ret = pinyin('中心', style=Style.FIRST_LETTER)
    print(ret)
    return


def demo4():
    """
    :return: [['zho1ng', 'zho4ng'], ['xi1n']]
    """
    ret = pinyin('中心', style=Style.TONE2, heteronym=True)
    print(ret)
    return


def demo5():
    """
    :return: [['zhong1', 'zhong4'], ['xin1']]
    """
    ret = pinyin('中心', style=Style.TONE3, heteronym=True)
    print(ret)
    return


def demo6():
    """
    不考虑多音字的情况. 没有声调.
    :return: ['zhong', 'xin']
    """
    ret = lazy_pinyin('中心')
    print(ret)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    # demo5()
    demo6()
