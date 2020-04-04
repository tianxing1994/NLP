"""
参考链接:
https://blog.csdn.net/say_c_box/article/details/78550659

数据集下载地址:
https://github.com/ningshixian/hmm-viterbi-Ch-POS

corpus_POS.txt:
存在这样的多重标注: [胜利/nz 海上/s 油田/n]nt 产/v 油/n 创/v 新高/n
split(sep=' ') 的结果为: ['[胜利/nz', '海上/s', '油田/n]nt', '产/v', '油/n', '创/v', '新高/n']
还可能有这种情况: [中国人民/n]/n
26个基本词类标记
（名词n、时间词t、处所词s、方位词f、数词m、量词q、区别词b、代词r、动词v、
形容词a、状态词z、副词d、介词p、连词c、助词u、语气词y、叹词e、拟声词o、
成语i、习惯用语l、简称j、前接成分h、后接成分k、语素g、非语素字x、标点符号w）
['v', 'n', 'u', 'a', 'w', 't', 'm', 'q', 'nt', 'nr', 'Vg',
'k', 'p', 'f', 'r', 'vn', 'ns', 'c', 's', 'd', 'ad', 'j', 'l',
'an', 'b', 'i', 'vd', 'z', 'nz', 'Ng', 'Tg', 'y', 'nx', 'Ag',
'o', 'Dg', 'Bg', 'h', 'Rg', 'vvn', 'e', 'Mg', 'na', 'Yg']
"""
from collections import defaultdict


def word_tag_split(word_tag):
    """
    示例:
    samples = ['[胜利/nz', '海上/s', '油田/n]nt', '产/v', '油/n', '创/v', '新高/n', '[中国人民/n]/n']
    for sample in samples:
        ret = word_tag_split(sample)
        print(ret)
    string = '[中国人民/n]/n'
    ret = string.split(sep=' ')
    print(ret)
    """
    l = len(word_tag)
    left = '['
    right = ']'
    start_idx = word_tag.find(left) + 1
    right_idx = word_tag.find(right)
    end_idx = right_idx if right_idx != -1 else l
    sub_string = word_tag[start_idx: end_idx]
    word, tag = sub_string.split(sep='/')
    return word, tag


def init_lambda():
    """
    lines: 计算行数, 以在最后, 计算初始状态概率 pi.
    :return:
    """
    pi = defaultdict(float)
    A = defaultdict(lambda: defaultdict(float))
    B = defaultdict(lambda: defaultdict(float))
    lines = 0
    cropus_pos_path = "../../dataset/part_of_speech_tagging_data/new_century/corpus_POS.txt"
    with open(cropus_pos_path, 'r', encoding='gbk') as f:
        for f_line_content in f:
            line_content = f_line_content.strip()
            if not line_content:
                continue
            word_tags = line_content.split(sep=' ')
            word, tag = word_tag_split(word_tags[0])
            pi[tag] += 1.
            B[tag][word] += 1.
            prev_tag = tag
            for word_tag in word_tags[1:]:
                word, tag = word_tag_split(word_tag)
                curr_tag = tag
                A[prev_tag][curr_tag] += 1.
                B[curr_tag][word] += 1.
                prev_tag = curr_tag
                lines += 1
            # break
        print(f"The accumulation of pi, A, B have been done!")
        print(f"length of pi: {len(pi)}")
        print(f"length of A: {len(A)}")
        print(f"length of B: {len(B)}")
        print(f"hidden state list: \n{sorted(B.keys())}")
        print(f"total lines: {lines}")
    for k, v in pi.items():
        pi[k] = v / lines
    for k0, sub_dict in A.items():
        total = sum(sub_dict.values())
        for k1, v in sub_dict.items():
            A[k0][k1] = v / total
    for k0, sub_dict in B.items():
        total = sum(sub_dict.values())
        for k1, v in sub_dict.items():
            B[k0][k1] = v / total
    print(f"Probability value of pi, A, B have been calculated!")
    return pi, A, B


def viterbi(obs, start_p, trans_p, emit_p):
    """
    :param obs: 可见序列
    :param start_p: 开始概率
    :param trans_p: 转换概率
    :param emit_p: 发射概率
    :return: 序列+概率
    """
    states = emit_p.keys()
    path = {}
    V = [{}]  # 记录第几次的概率
    for state in states:
        V[0][state] = start_p[state] * emit_p[state].get(obs[0], 0)
        path[state] = [state]
    for n in range(1, len(obs)):
        V.append({})
        newpath = {}
        for k in states:
            pp,pat=max([(V[n - 1][j] * trans_p[j].get(k,0) * emit_p[k].get(obs[n], 0) ,j )for j in states])
            V[n][k] = pp
            newpath[k] = path[pat] + [k]
            # path[k] = path[pat] + [k]#不能提起变，，后面迭代好会用到！
        path=newpath
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return prob, path[state]


def demo1():
    pi, A, B = init_lambda()
    test_strs = [['你们', '站立', '在'],
                 ['我', '站', '在', '北京', '天安门', '上', '大声', '歌唱'],
                 ['请', '大家', '坐下', '喝茶'],
                 ['你', '的', '名字', '是', '什么'],
                 ['今天', '天气', '特别', '好']]
    for line in test_strs:
        print(line)
        p, out_list = viterbi(line, pi, A, B)
        print(p, out_list)
    return


if __name__ == '__main__':
    demo1()
