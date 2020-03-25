import jieba
import jieba.analyse


def demo1():
    """
    jieba.cut 中文分词
    """
    sent = "在包含问题的所有解的解空间树中, 按照深度优先搜索的策略, 从根节点出发深度探索解空间树. "

    # word_iterator = jieba.cut(sent, cut_all=False)
    # ['在', '包含', '问题', '的', '所有', '解', '的', '解', '空间', '树中', ',', ' ', '按照', '深度', '优先', '搜索',
    # '的', '策略', ',', ' ', '从根', '节点', '出发', '深度', '探索', '解', '空间', '树', '.', ' ']

    word_iterator = jieba.cut(sent, cut_all=True)
    # ['在', '包含', '问题', '的', '所有', '解', '的', '解空', '空间', '树', '中', ',', ' ', '', '按照', '深度', '优先',
    # '搜索', '的', '策略', ',', ' ', '', '从', '根', '节点', '点出', '出发', '深度', '探索', '索解', '解空', '空间', '树', '.', '', ' ', '']

    print(list(word_iterator))
    return


def demo2():
    sent = "在包含问题的所有解的解空间树中, 按照深度优先搜索的策略, 从根节点出发深度探索解空间树. "

    word_iterator = jieba.cut_for_search(sent)
    print(list(word_iterator))
    return


def demo3():
    """
    增加用户词典后的中文分词.
    可以更好地处理专业名词.

    词典如下:
    解空间 5 n
    解空间树 5 n
    根结点 5 n
    深度优先 5 n

    词语, 词频, 词性. 词频用于设置词的权重, 越大则越不容易受到内置词典的干扰.
    """
    jieba.load_userdict("../dataset/jieba_dataset/userdict.txt")
    sent = "在包含问题的所有解的解空间树中, 按照深度优先搜索的策略, 从根节点出发深度探索解空间树. "
    word_iterator = jieba.cut(sent, cut_all=False)
    # ['在', '包含', '问题', '的', '所有', '解', '的', '解空间树', '中', ',', ' ', '按照', '深度优先', '搜索', '的',
    # '策略', ',', ' ', '从根', '节点', '出发', '深度', '探索', '解空间树', '.', ' ']

    print(list(word_iterator))
    return


def demo4():
    sent = "在包含问题的所有解的解空间树中, 按照深度优先搜索的策略, 从根节点出发深度探索解空间树. "

    # word_list = jieba.cut(sent, cut_all=False)
    # word_list = jieba.lcut(sent, cut_all=True)
    word_list = jieba.lcut_for_search(sent, HMM=True)

    print(word_list)
    return


def demo5():
    sent = '如果放在 post 中将出错. '

    word_list = jieba.lcut(sent, HMM=False)
    print(word_list)

    jieba.suggest_freq(('中', '将'), True)
    word_list = jieba.lcut(sent, HMM=False)
    print(word_list)
    return


def demo6():
    """
    停用词
    """
    sent = '今天天气不错'
    word_list = jieba.lcut(sent, HMM=False)
    print(word_list)

    jieba.del_word('今天天气')
    word_list = jieba.lcut(sent, HMM=False)
    print(word_list)

    return


def demo7():
    """
    Tokenize: 返回词语在原文的起止位置
    """
    print(' 默认模式')
    result = jieba.tokenize('永和服装饰品有限公司')
    for tk in result:
        print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))

    print(' 搜索模式')
    result = jieba.tokenize('永和服装饰品有限公司', mode='search')
    for tk in result:
        print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))
    return


def demo8():
    """
    paddle 模式: 利用 PaddlePaddle 深度学习框架, 训练序列标注 (双向 GRU) 网络模型实现分词.
    同时支持词性标注. (应该是百度提供的分词模型). pip install paddlepaddle-tiny==1.6.1
    """
    content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
    # use_paddle: 是否使用 paddle 模式下的分词模式.
    word_iterator = jieba.cut(content, use_paddle=True)
    print(list(word_iterator))
    return


if __name__ == '__main__':
    demo8()
