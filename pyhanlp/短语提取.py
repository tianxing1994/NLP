from pyhanlp import HanLP


text = "在计算机音视频和图形图像技术等二维信息算法处理方面目前比较先进的视频处理算法"
phraseList = HanLP.extractPhrase(text, 10)
print(phraseList)
