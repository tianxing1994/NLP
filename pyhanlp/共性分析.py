from pyhanlp import JClass


# 共性分析
Occurrence = JClass("com.hankcs.hanlp.corpus.occurrence.Occurrence")
PairFrequency = JClass("com.hankcs.hanlp.corpus.occurrence.PairFrequency")
TermFrequency = JClass("com.hankcs.hanlp.corpus.occurrence.TermFrequency")
TriaFrequency = JClass("com.hankcs.hanlp.corpus.occurrence.TriaFrequency")

occurrence = Occurrence()
occurrence.addAll("在计算机音视频和图形图像技术等二维信息算法处理方面目前比较先进的视频处理算法")
occurrence.compute()

print("一阶共性分析，也就是词频统计")
unigram = occurrence.getUniGram()
for entry in unigram.iterator():
    term_frequency = entry.getValue()
    print(term_frequency)
print()

print('二阶共性分析')
bigram = occurrence.getBiGram()
for entry in bigram.iterator():
    pair_frequency = entry.getValue()
    if pair_frequency.isRight():
        print(pair_frequency)
print()

print('三阶共性分析')
trigram = occurrence.getTriGram()
for entry in trigram.iterator():
    tria_frequency = entry.getValue()
    if tria_frequency.isRight():
        print(tria_frequency)
