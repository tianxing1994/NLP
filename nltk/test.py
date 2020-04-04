import nltk


sent = nltk.corpus.treeback.tagged_sents()[22]
print(sent)
ret = nltk.ne_chunk(sent, binary=True)
print(ret)

