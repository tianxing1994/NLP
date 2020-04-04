from pyhanlp import JClass


def demo1():
    """
    CRF词法分析器

    返回值:
    [上海/ns 华安/nz 工业/n （/w 集团/n ）/w 公司/n]/nt 董事长/n 谭旭光/nr 和/c 秘书/n 胡花蕊/nr 来到/v [美国纽约/ns 现代/ntc 艺术/n 博物馆/n]/ns 参观/v
    """
    CRFLexicalAnalyzer = JClass("com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")
    analyzer = CRFLexicalAnalyzer()

    sentence = "上海华安工业（集团）公司董事长谭旭光和秘书胡花蕊来到美国纽约现代艺术博物馆参观"
    ret = analyzer.analyze(sentence)
    print(ret)
    return


if __name__ == "__main__":
    demo1()
