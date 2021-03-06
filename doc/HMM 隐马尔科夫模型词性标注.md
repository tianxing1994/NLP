
## HMM 隐马尔科夫模型词性标注
https://blog.csdn.net/say_c_box/article/details/78550659
https://github.com/ningshixian/hmm-viterbi-Ch-POS


隐马尔科夫模型是结构最简单的动态贝叶斯网络. 描述由一个隐藏的马尔科夫链随机生成不可观测的状态随机序列, 再由各个状态生成一个观测而产生随机序列的过程. 隐藏的马尔科夫链随机生成的状态的序列称为状态序列, 每个状态生成一个观测, 称为观测序列.


隐马尔科夫做了两个基本的假设.

* 齐次马尔科夫假设, 即假设隐藏的马尔科夫链在任意时刻 $t$ 的状态只依赖于前一时刻的状态, 去其他观测状态无关.
* 观测独立性假设, 即假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态, 与其他观测以及状态无关.
  隐马尔科夫模型由初始状态概率向量 $\pi$, 状态转移概率矩阵 $A$, 以及观测概率矩阵 $B$ 决定.



在词性标注问题中:

* 初始状态概率: 为每个语句序列开头出现的词性的概率,
* 状态转移概率矩阵$A$: 前一次是某个状态, 而这一次转换为各个状态的概率矩阵. 由相邻两个单词的词性得到.
*  观测转移概率矩阵$B$: 当前的状态转换输出的各观测值的概率矩阵. 由词性输出各种词的概率得出.
  观测序列为分词后的单词序列, 状态序列为每个单词的词性






隐马尔科夫模型有三个基本问题:

* 概率计算问题, 给出模型和观测序列, 计算在模型 $\lambda = (A, B, \pi)$ 下观测序列 $O$ 出现的概率.
* 学习问题, 估计模型 $\lambda = (A, B, \pi)$ 参数, 使得该模型下观测序列 $P(O \vline \lambda)$ 最大, 也就是用极大似然的方法估计参数.
* 观测问题, 已知模型 $\lambda$ 和观测序列 $O$, 求对观测序列条件概率 $P(I \vline O)$ 最大的状态序列 $I$, 即从观测序列, 求最可能的状态序列.



在词性标注问题中, 需要解决的是学习问题和观测问题. 学习问题即转移矩阵的构建, 观测问题即根据单词序列得到对应的词性标注序列.



### 学习问题
三个概率的计算是该算法的核心.

* 转移概率 $a_{ij}$ 的计算.
  $$\begin{aligned} a_{ij} = \frac{A_{ij}}{\sum_{j=1}^{N}A_{ij}} \end{aligned}$$

  其中: $A_{ij}$ 表示从 $t$ 时刻到 $t+1$ 时刻, 从状态 $i$ 变成状态 $j$ 的频数.

* 观测概率 $b_{j}(k)$ 的计算.
  $$\begin{aligned} b_{j}(k) = \frac{B_{jk}}{\sum_{k=1}^{M}B_{jk}} \end{aligned}$$
  其中 $B_{jk}$ 表示状态为 $j$, 观测为 $k$ 的概率.


* 初始状态概率 $\pi_{i}$ 的计算. 估计为多个序列中, 初始状态 $q_{i}$ 出现的频率.




### 观测问题
通常采用维特比算法来解决观测问题. 本质上是一个非常简单的动态规划问题. 利用 $dp_{ij}$ 表示处理到第 $i$ 个单词, 该单词词性为 $j$ 的序列出现的概率. 由于做了齐次马尔科夫的假设, 所以这样做是不会有后效性的. 转移到 $dp_{i+1}$ 也十分简单.





