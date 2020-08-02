#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pycorrector


def demo1():
    corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
    print(corrected_sent, detail)
    return


def demo2():
    maybe_errors = pycorrector.detect('少先队员因该为老人让坐')
    print(maybe_errors)
    return


if __name__ == '__main__':
    demo2()
