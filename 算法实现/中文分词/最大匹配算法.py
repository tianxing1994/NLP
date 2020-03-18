"""
参考链接:
https://www.cnblogs.com/Jm-15/p/9403352.html

训练语料和测试语料下载:
链接: https://pan.baidu.com/s/1X0coEznut6_s0jsDG9_9Dg 密码: b393

### 优点
在已有相同类型的词汇集时, 可以比较好地用于目标分词. 算法简单, 易于实现.

### 正向最大匹配算法:

例如: 词典中最长词有 7 个汉字, 则最大匹配起始字数为 7 个汉字. 然后逐字递减, 在对应的词典中进行查找.

以 "我们在野生动物园玩" 为例说明正向最大匹配方法:

1, 正向最大匹配法:
正向即从前往后取词, 从 7->1, 每次减一个字, 直到词典命中或剩下 1 个单字.

第 1 轮扫描:
第 1 次: "我们在野生动物", 扫描 7 字词典, 无
第 2 次: "我们在野生动", 扫描 6 字词典, 无
......
第 6 次: "我们", 扫描 2 字词典, 有

扫描中止, 输出第 1 个词为 "我们", 去除第 1 个词后开始第 2 轮扫描, 即:

第 2 轮扫描:
第 1 次: "在野生动物园玩", 扫描 7 字词典, 无
第 2 次: "在野生动物园", 扫描 6 字词典, 无
......
第 6 次: "在野", 扫描 2 字词典, 有

扫描中止, 输出第 2 个词为 "在野", 去除第 2 个词后开始第 3 轮扫描, 即:

第 3 轮扫描:
第 1 次: "生动物园玩", 扫描 5 字词典, 无
第 2 次: "生动物园", 扫描 4 字词典, 无
第 3 次: "生动物", 扫描3字词典, 无
第 4 次: "生动", 扫描 2 字词典, 有

扫描中止, 输出第 3 个词为 "生动", 第 4 轮扫描, 即:

第 4 轮扫描:
第 1 次: "物园玩", 扫描 3 字词典, 无
第 2 次: "物园", 扫描 2 字词典, 无
第 3 次: "物", 扫描 1 字词典, 无

扫描中止, 输出第 4 个词为 "物", 非字典词数加 1, 开始第 5 轮扫描, 即:

第 5 轮扫描:
第 1 次: "园玩", 扫描 2 字词典, 无
第 2 次: "园", 扫描 1 字词典, 有

扫描中止, 输出第 5 个词为 "园", 单字字典词数加 1, 开始第 6 轮扫描, 即:

第 6 轮扫描:
第 1 次: "玩", 扫描 1 字典词, 有

扫描中止, 输出第 6 个词为 "玩", 单字字典词数加 1, 整体扫描结束.

正向最大匹配法, 最终切分结果为: "我们/在野/生动/物/园/玩".



### 逆向最大匹配算法:
逆向即从后往前取词, 其他逻辑和正向相同.

第 1 轮扫描: "在野生动物园玩"

第 1 次: "在野生动物园玩", 扫描 7 字词典, 无
第 2 次: "野生动物园玩", 扫描 6 字词典, 无
......
第 7 次: "玩", 扫描 1 字词典, 有

扫描中止, 输出 "玩", 单字字典词加 1, 开始第 2 轮扫描

第 2 轮扫描: "们在野生动物园"

第 1 次: "们在野生动物园", 扫描 7 字词典, 无
第 2 次: "在野生动物园", 扫描 6 字词典, 无
第 3 次: "野生动物园", 扫描 5 字词典, 有

扫描中止, 输出 "野生动物园", 开始第 3 轮扫描

第 3 轮扫描: "我们在"

第 1 次: "我们在", 扫描 3 字词典, 无
第 2 次: "们在", 扫描 2 字词典, 无
第 3 次: "在", 扫描 1 字词典, 有

扫描中止, 输出 "在", 单字字典词加 1, 开始第 4 轮扫描

第 4 轮扫描: "我们"

第 1 次: "我们", 扫描 2 字词典, 有

扫描中止, 输出 "我们", 整体扫描结束.

逆向最大匹配法. 最终切分结果为: "我们/在/野生动物园/玩"
"""


def get_dictionary(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            file_content = f.read().split(sep='  ')
        finally:
            f.close()
        chars = list(set(file_content))
    return chars


def forward_max_matching(src_string, dictionary, max_length=5):
    """
    正向最大匹配算法
    """
    src_length = len(src_string)
    ret = list()

    start_idx = 0
    valid_len = min(max_length, src_length)
    end_idx = valid_len - 1

    while True:
        try_word = src_string[start_idx: end_idx]
        if (try_word in dictionary) or (end_idx - start_idx == 1):
            ret.append(try_word)
            start_idx = end_idx
            end_idx = min(end_idx + max_length, src_length)
            if start_idx == end_idx:
                break
        else:
            end_idx -= 1
    return ret


def backward_max_matching(src_string, dictionary, max_length=5):
    """
    逆向最大匹配算法
    """
    src_length = len(src_string)
    ret = list()

    end_idx = src_length
    valid_len = min(max_length, src_length)
    start_idx = end_idx - valid_len

    while True:
        try_word = src_string[start_idx: end_idx]
        if (try_word in dictionary) or (end_idx - start_idx == 1):
            ret.append(try_word)
            end_idx = start_idx
            start_idx = max(start_idx - max_length, 0)
            if start_idx == end_idx:
                break
        else:
            start_idx += 1
    ret.reverse()
    return ret


def calc_score(src_list, dst_list):
    """
    计算分词与目标之间的得分, 可以分出词的正确个数除以目标个数 (召回率), 也可以计算分出正确词的个数除以分出词的总数 (精确度).
    """
    dst_words_num = len(dst_list)
    src_words_num = len(src_list)
    src_max_idx = len(src_list) - 1
    dst_max_idx = len(dst_list) - 1
    src_idx, dst_idx = 0, 0
    correct_num = 0
    while (src_idx <= src_max_idx) and (dst_idx <= dst_max_idx):
        src_word = src_list[src_idx]
        dst_word = dst_list[dst_idx]

        if src_word == dst_word:
            correct_num += 1
        else:
            while (src_idx <= src_max_idx) and (dst_idx <= dst_max_idx):
                src_len = len(src_word)
                dst_len = len(dst_word)
                if src_len > dst_len:
                    dst_idx += 1
                    dst_word += dst_list[dst_idx]
                elif src_len < dst_len:
                    src_idx += 1
                    src_word += src_list[src_idx]
                else:
                    break
        src_idx += 1
        dst_idx += 1
    recall = correct_num / dst_words_num
    accuracy = correct_num / src_words_num
    return accuracy, recall, correct_num, src_words_num, dst_words_num


def demo1():
    """
    正向最大匹配算法演示.
    """
    train_file = "../../dataset/max_matching_corpus/train.txt"
    src_string = "共同创造美好的新世纪——二○○一年新年贺词"

    dictionary = get_dictionary(train_file)
    result = forward_max_matching(src_string, dictionary, max_length=5)
    print(result)
    return


def demo2():
    """
    逆向最大匹配算法演示.
    """
    train_file = "../../dataset/max_matching_corpus/train.txt"
    src_string = "共同创造美好的新世纪——二○○一年新年贺词"
    dictionary = get_dictionary(train_file)
    result = backward_max_matching(src_string, dictionary, max_length=5)
    print(result)
    return


def demo3():
    """
    计算分词的得分.
    """
    src_list = ['共同', '创造', '美好', '的', '新世纪', '——', '二', '○', '○', '一', '年', '新年', '贺词']
    dst_list = ['共同', '创造', '美好', '的', '新', '世纪', '——', '二○○一年', '新年', '贺词']
    ret = calc_score(src_list, dst_list)
    print(ret)
    return


def demo4():
    """
    测试分词的得分.
    """
    train_file = "../../dataset/max_matching_corpus/train.txt"
    test_file = "../../dataset/max_matching_corpus/test.txt"
    target_file = "../../dataset/max_matching_corpus/test_gold.txt"
    with open(test_file, 'r', encoding='utf-8') as f:
        src_string_list = f.readlines()
    with open(target_file, 'r', encoding='utf-8') as f:
        dst_string_list = f.readlines()

    total_correct = 0
    total_dst_words_num = 0

    dictionary = get_dictionary(train_file)
    for src_string, dst_string in zip(src_string_list, dst_string_list):
        src_string = src_string.strip()
        dst_string = dst_string.strip()

        src_list = forward_max_matching(src_string, dictionary, max_length=5)
        dst_list = dst_string.split(sep='  ')
        print("src_list: ", src_list)
        print("dst_list: ", dst_list)
        accuracy, recall, correct_num, src_words_num, dst_words_num = calc_score(src_list, dst_list)
        print(accuracy, recall, correct_num, src_words_num, dst_words_num)
        total_correct += correct_num
        total_dst_words_num += dst_words_num
    total_recall = total_correct / total_dst_words_num
    print(total_recall)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    demo4()

