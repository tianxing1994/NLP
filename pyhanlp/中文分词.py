from pyhanlp import HanLP, JClass


def demo1():
    sentence = "下雨天地面积水"

    # 返回一个list，每个list是一个分词后的Term对象，可以获取word属性和nature属性，分别对应的是词和词性
    terms = HanLP.segment(sentence)
    for term in terms:
        print(term.word, term.nature)

    return


def demo2():
    """
    词性对照表.
    https://github.com/hankcs/HanLP/blob/master/data/dictionary/other/TagPKU98.csv
    :return:
    """
    NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")

    sent1 = "我新造一个词叫幻想乡你能识别并正确标注词性吗？"
    ret = NLPTokenizer.segment(sent1)
    print(ret)

    sent2 = "我的希望是希望张晚霞的背影被晚霞映红"
    ret = NLPTokenizer.analyze(sent2)
    print(ret)
    print(ret.translateLabels())

    sent3 = "支援臺灣正體香港繁體：微软公司於1975年由比爾·蓋茲和保羅·艾倫創立"
    ret = NLPTokenizer.analyze(sent3)
    print(ret)
    print(ret.translateLabels())
    return


def demo3():
    """N-最短路径分词. 该分词器比最短路分词器慢, 但是效果稍微好一些, 对命名实体识别能力更强"""
    sentences = ["今天，刘志军案的关键人物,山西女商人丁书苗在市二中院出庭受审。",
                 "江西省监狱管理局与中国太平洋财产保险股份有限公司南昌中心支公司保险合同纠纷案",
                 "新北商贸有限公司"]

    # N-最短路径分词
    NShortSegment = JClass("com.hankcs.hanlp.seg.NShort.NShortSegment")
    nshort_segment = NShortSegment().\
        enableCustomDictionary(False).\
        enablePlaceRecognize(True).\
        enableOrganizationRecognize(True)

    for sentence in sentences:
        ret = nshort_segment.seg(sentence)
        print(ret)

    # 最短路分词
    ViterbiSegment = JClass("com.hankcs.hanlp.seg.Viterbi.ViterbiSegment")
    shortest_segment = ViterbiSegment().\
        enableCustomDictionary(False).\
        enablePlaceRecognize(True).\
        enableOrganizationRecognize(True)

    for sentence in sentences:
        ret = shortest_segment.seg(sentence)
        print(ret)
    return


def demo4():
    """
    演示URL识别
    输出结果:
    [HanLP/nx, 的/ude1, 项目/n, 地址/n, 是/vshi, https://github.com/hankcs/HanLP/xu, ，/w,
    /w, 发布/v, 地址/n, 是/vshi, https://github.com/hankcs/HanLP/releases/xu, ，/w,
    /w, 我/rr, 有时候/d, 会/v, 在/p, www/nx, ./w, hankcs/nrf, ./w, com/nx, 上面/f, 发布/v, 一些/m, 消息/n, ，/w,
    /w, 我/rr, 的/ude1, 微博/n, 是/vshi, http://weibo.com/hankcs/xu, //w, ，/w,
    /w, 会/v, 同步/vd, 推送/nz, hankcs/nrf, ./w, com/nx, 的/ude1, 新闻/n, 。/w, 听说/v, ./w, 中国/ns, ,/w, 因为/c, 穷/a, ……/w,
    /w]
    https://github.com/hankcs/HanLP
    https://github.com/hankcs/HanLP/releases
    http://weibo.com/hankcs
    """
    text = '''HanLP的项目地址是https://github.com/hankcs/HanLP，
    发布地址是https://github.com/hankcs/HanLP/releases，
    我有时候会在www.hankcs.com上面发布一些消息，
    我的微博是http://weibo.com/hankcs/，
    会同步推送hankcs.com的新闻。听说.中国,因为穷……
    '''
    URLTokenizer = JClass("com.hankcs.hanlp.tokenizer.URLTokenizer")
    term_list = URLTokenizer.segment(text)
    print(term_list)

    Nature = JClass("com.hankcs.hanlp.corpus.tag.Nature")

    for term in term_list:
        # 如果 term 的词性为 Nature.xu 词性. 打印.
        if term.nature == Nature.xu:
            print(term.word)
    return


def demo5():
    """
    基础分词,
    基础分词只进行基本 NGram 分词, 不识别命名实体, 不使用用户词典.
    n-gram 语言模型:
    一个 item 的出现概率, 只与其前 m 个 items 有关, 当 m=0 时, 就是 unigram, m=1 时, 是 bigram 模型.
    参考链接:
    https://blog.csdn.net/ahmanz/article/details/51273500
    """
    text = ("举办纪念活动铭记二战历史，不忘战争带给人类的深重灾难，是为了防止悲剧重演，确保和平永驻；"
            "铭记二战历史，更是为了提醒国际社会，需要共同捍卫二战胜利成果和国际公平正义，"
            "必须警惕和抵制在历史认知和维护战后国际秩序问题上的倒行逆施。")
    BasicTokenizer = JClass("com.hankcs.hanlp.tokenizer.BasicTokenizer")
    ret = BasicTokenizer.segment(text)
    print(ret)
    return


def demo6():
    """演示用户词典的动态增删"""
    text = "攻城狮逆袭单身狗，迎娶白富美，走上人生巅峰"
    print(HanLP.segment(text))

    CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
    CustomDictionary.add("攻城狮")  # 动态增加
    CustomDictionary.insert("白富美", "nz 1024")  # 强行插入
    # CustomDictionary.remove("攻城狮")
    CustomDictionary.add("单身狗", "nz 1024 n 1")
    # print(CustomDictionary.get("单身狗"))

    print(HanLP.segment(text))


if __name__ == '__main__':
    # demo1()
    demo2()
