## Word2Vec 词向量

https://www.jianshu.com/p/b779f8219f74

假设文档中有 100 个不同的词, 则每可以用一个 100 维的向量来表示这一句个词, 每一个向量上都只有一个维度的值为 1, 其它的值为 0 (独热编码). 

独热编码的方式所使用的维度太大, 且每个维度的取值只有 1 或 0, 这存在空间的浪费. 且这种编码方式不能体现词与词之间的相似性, 即, 不能通过向量的相关性来体现词的相关性. 

所以希望将这种稀疏向量映射成稠密向量 (向量的维度减少, 向量每个维度的值可以是任意数值而不仅是 1 或 0), 希望映射产生的稠密向量能够体现词与词的相似性. 

设输入向量 input_vector  是 (n, 1) 维的独热向量, 对应的输出向量 output_vector 是 (m, 1) 维的稠密向量. 显然, 只需要一个形状为 (m, n) 转换矩阵 $M_{m \times n}$ 就可以将 input_vector 映射为 output_vector . 再需要一个形状为 (n, m) 的转换矩阵 $N_{n \times m}$ 就可以将 output_vector 映射为 input_vector 向量. 如下: 

$$\begin{aligned} M_{m \times n} \cdot I_{n \times1} &= O_{m \times 1} \\ N_{n \times m} \cdot O_{m \times 1} &= \hat{I}_{n \times 1} \\ softmax(\hat{I}_{n \times 1}) &= I_{prob} \\ argmax(I_{prob}) &= I_{n \times 1} \end{aligned}$$ 



注意到: $M_{m \times n} \cdot I_{n \times1} = O_{1 \times m}$ 中, 因为 $I_{n \times 1}$ 向量中只有一个值为 1 其它都为 0. 所以 $O_{1 \times m}$ 实际上是矩阵 $M_{m \times n}$ 中的某一列的值. 而这一列的值在与矩阵 $N_{n \times m}$ 运算时, 刚好只与其中某一行的值为的内积结果为 1, 与其它行的结果为 0. 

以上方法训练出来的词向量 $O_{m \times 1}$ 实际上只是减少了词向量 $I_{n \times 1}$ 的维数. 它还不能够体现词与词之间的相似性. 

以下介绍 skip-gram 和 CBOW 模型. 



### skip-gram 模型

skip-gram 的思想是根据中心词来预测周围的词. 假设中心词是 cat, 窗口长度为 2, 则根据 cat 预测左边两个词和右边两个词. 这时, cat 作为神经网络的输入 input, 预测的词作为 label. 

如句子: the quick brown fox jumps over the lazy dog. 可以产生训练样本: 

(the, quick), (the, brown); (quick, the), (quick, brown), (quick, fox); (brown, the), (brown, quick), (brown, fox), (brown, jumps); (fox, quick), (fox, brown), (fox, jumps), (fox, over), ... 等. 

通过前面讲述的词向量模型. 当输入表示词 the 的独热向量 $I_{n \times 1}^{the}$ 时, 其经过 $M$ 和 $N$ 矩阵及 $Softmax$ 处理后得到独热向量 $I_{n \times 1}^{quick}$  , 表示如下: 

$$\begin{aligned} M_{m \times n} \cdot I_{n \times 1}^{the} &= O_{m \times 1}^{the} \\ N_{n \times m} \cdot O_{m \times 1}^{the} &= \hat{I}_{n \times 1} \\ softmax(\hat{I}_{n \times 1}) &= I_{n \times 1}^{prob} \\ argmax(I_{n \times 1}^{prob}) &= I_{n \times 1}^{quick} \end{aligned}$$ 

注意到训练样本有: (the, quick), (the, brown); 也就是说, 词 the, 既要得到 quick, 又要得到 brown, 这似乎是矛盾的. 可以想到, 神经网络训练时, 上式中 $\hat{I}$ 向量对应于词 the, quick 的独热向量的两个值会比其它值都大, 也就是训练时这两个维度的值会取得一个平衡, 使得损失最小. 

考虑到更一般的情况, 则 $I_{n \times 1 }^{prob}$ 向量中各维度的值的大小也可以看作是词 the 周围出现其它各个词的概率. 相似的两个词, 最终获得的 $I_{n \times 1}^{prob}$ 向量之间必定具有较大的相关性. 同样地, 同似词的稠密向量 $O$ 也具有较大的相关性 (我猜的, 稠密向量的相关性应由 $I_{n \times 1}^{prob}$ 向量的相关性推导过来, 我暂时推不出来). 



### CBOW (continuous-bag-of_words) 模型

CBOW 模型与 skip-gram 模型刚好相反, 它是用周围的词来预测中心词. 

 如句子: the quick brown fox jumps over the lazy dog. 可以产生训练样本: 

([quick, brown], the); ([the, brown, fox], quick); ([the, quick, fox, jumps], brown); ([quick, brown, jumps, over], fox);... 等. 

这时, input 输入很可能是 4 个词, label 只是一个词. 计算时, 只要对其求平均, 经过隐藏层后, 输入的 4 个词被映射成 4 个 n 维向量.  对这 4 个向量求平均, 就可以作为下一层的输入. 表示如下: 

$$\begin{aligned} M_{m \times n} \cdot I_{n \times 4} &= O_{m \times 4} \\ average(O_{m \times 4}) &= O_{m \times 1} \\ N_{n \times m} \cdot O_{m \times 1} &= \hat{I}_{n \times 1} \\ softmax(\hat{I}_{n \times 1}) &= I_{n \times 1}^{prob} \\ argmax(I_{n \times 1}^{prob}) &= I_{n \times 1} \end{aligned}$$ 

同样地,  $I_{n \times 1 }^{prob}$ 向量中各维度的值的大小可以看作是周围词出现时, 其中心词的概率. 



skip-gram, CBOW 两模型相比, skip-gram 模型能产生更多训练样本, 抓住更多词与词之间语义上的细节, 在语料足够多足够好的理想条下, skip-gram 模型是优于 CBOW 模型的. 但在语料较少的情况下, 其难以抓住足够多词与词之间的细节, CBOW 模型求平均的特性, 反而效果可能更好. 





### 负采样 (Negative Sampling)

实际训练时, 假设词库有 10000 个词, 而目标词向量为 300 维, 那么转换矩阵 $M_{n \times m}$ 和 $N_{m \times n}$ 就有 $n \times m = 10000 \times 300 = 3000000$ 个参数. 这样简单的一层网络就有三百万个参数. 其计算量是非常大的. 

为了减少计算量, 在神经网络模型计算损失时, 我们只从 10000 个维度中随机选择一部分维度 (其中一定包含 target 目标中值为 1 的维度) 进行参数更新. 这样一来, 一次更新所需要计算的参数量就大大减少了. 



论文中, 每个维度随机选择的权重通过下式计算. 

$$\begin{aligned} P(w_{i}) = \frac{f(w_{i})^{3/4}}{\sum_{j=0}^{n}{(f(w_{j})^{3/4})}} \end{aligned}$$ 





















