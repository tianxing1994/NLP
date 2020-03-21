import jieba.analyse


def demo1():
    """
    基于 TF-IDF 算法的关键词抽取.
    """
    content = "在包含问题的所有解的解空间树中, 按照深度优先搜索的策略, 从根节点出发深度探索解空间树. "

    tags = jieba.analyse.extract_tags(content, topK=5)
    print(tags)
    return


if __name__ == '__main__':
    demo1()
