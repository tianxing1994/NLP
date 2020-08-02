from nltk.tokenize import TweetTokenizer

tk = TweetTokenizer()

fpath = 'test.enc'
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        print(line)

        geek = tk.tokenize(line)
        print(geek)
