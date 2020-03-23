import jieba.analyse


def demo1():
    """
    基于 TF-IDF 算法的关键词抽取.
    """
    content = "在包含问题的所有解的解空间树中, 按照深度优先搜索的策略, 从根节点出发深度探索解空间树. "
    tags = jieba.analyse.extract_tags(content, topK=5)
    print(tags)

    content = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，" \
              "吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。" \
              "目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
    tags = jieba.analyse.extract_tags(content, withWeight=True)
    print(tags)
    return


def demo2():
    content = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，" \
              "吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。" \
              "目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
    tags = jieba.analyse.textrank(content, withWeight=True)
    print(tags)
    return


if __name__ == '__main__':
    demo1()
