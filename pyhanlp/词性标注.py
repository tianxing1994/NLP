from pyhanlp import JClass, HanLP


def demo1():
    """ 演示自定义词性,以及往词典中插入自定义词性的词语
        !!!由于采用了反射技术,用户需对本地环境的兼容性和稳定性负责!!!
    TO-DO
    如果使用了动态词性之后任何类使用了switch(nature)语句,必须注册每个类
    """
    # 对于系统中已有的词性,可以直接获取
    Nature = JClass("com.hankcs.hanlp.corpus.tag.Nature")
    pc_nature = Nature.fromString("n")
    print(pc_nature)
    # 此时系统中没有"电脑品牌"这个词性
    pc_nature = Nature.fromString("电脑品牌")
    print(pc_nature)
    # 我们可以动态添加一个
    pc_nature = Nature.create("电脑品牌")
    print(pc_nature)
    # 可以将它赋予到某个词语
    LexiconUtility = JClass("com.hankcs.hanlp.utility.LexiconUtility")
    LexiconUtility.setAttribute("苹果电脑", pc_nature)
    # 或者
    LexiconUtility.setAttribute("苹果电脑", "电脑品牌 1000")
    # 它们将在分词结果中生效
    term_list = HanLP.segment("苹果电脑可以运行开源阿尔法狗代码吗")
    print(term_list)
    for term in term_list:
        if term.nature == pc_nature:
            print("找到了 [{}] : {}\n".format(pc_nature, term.word))

    # 还可以直接插入到用户词典
    CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
    CustomDictionary.insert("阿尔法狗", "科技名词 1024")
    StandardTokenizer = JClass("com.hankcs.hanlp.tokenizer.StandardTokenizer")
    StandardTokenizer.SEGMENT.enablePartOfSpeechTagging(True)  # 依然支持隐马词性标注
    term_list = HanLP.segment("苹果电脑可以运行开源阿尔法狗代码吗")
    print(term_list)
    return


if __name__ == '__main__':
    demo1()
