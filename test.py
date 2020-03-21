import jieba

sent = "在包含问题的所有解的解空间树中, 按照深度优先搜索的策略, 从根节点出发深度探索解空间树. "
word_list = jieba.cut_for_search(sent)
print(list(word_list))
