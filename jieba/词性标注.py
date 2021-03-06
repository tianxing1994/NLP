import jieba.posseg as pseg


def demo1():
    sent = "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n" \
           "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n" \
           "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"

    ret = pseg.cut(sent)

    print(list(ret))
    return


def demo2():
    sent = "程序员祝海林和朱会震是在孙健的左面和右面, 范凯在最右面.再往左是李松洪"
    ret = pseg.cut(sent, use_paddle=True)
    print(list(ret))
    return


if __name__ == '__main__':
    # demo1()
    demo2()
