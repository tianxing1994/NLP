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
    content = "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n" \
              "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n" \
              "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"

    jieba.analyse.set_idf_path("../dataset/jieba_dataset/extra_dict/idf.txt.big")

    tags = jieba.analyse.extract_tags(content, topK=10, withWeight=True)
    print(tags)
    return


def demo3():
    content = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，" \
              "吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。" \
              "目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
    tags = jieba.analyse.textrank(content, withWeight=True)
    print(tags)
    return


def demo4():
    """
    可以看到 textrank 得到的结果中会有 "认为", "看来", "结果" 等无用的词. 而 extract_tags 就要好很多.
    认为, textrank 是在数据量足够大时, 表现的效果会比较好.

    输出结果:
    ['全明星赛', '勇士', '正赛', '指导', '对方', '投篮', '球员',
    '没有', '出现', '时间', '威少', '认为', '看来', '结果', '相隔',
    '助攻', '现场', '三连庄', '介绍', '嘉宾']

    ['韦少', '杜兰特', '全明星', '全明星赛', 'MVP', '威少', '正赛',
    '科尔', '投篮', '勇士', '球员', '斯布鲁克', '更衣柜', 'NBA', '三连庄',
    '张卫平', '西部', '指导', '雷霆', '明星队']
    """
    nba = "../dataset/others/documents/NBA.txt"
    with open(nba, 'r', encoding='utf-8') as f:
        nba_content = f.read()
    tags = jieba.analyse.textrank(nba_content)
    print(tags)
    tags = jieba.analyse.extract_tags(nba_content)
    print(tags)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
