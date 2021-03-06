## 基于线性链 CRF 的 Web 观点挖掘学习方法

http://www.comp.hkbu.edu.hk/~lichen/download/WISE10_QiChen.pdf

```text
备注: 
这是基于手工特征的算法. 对于句子中的每个位置, 根据词与词性手工为当前位置设置多个特征(每种特征的权重 lambda 不同), 任意一个特征满足时为该位置的特定标签加权, 之后将权重转换为概率. 最后可通过 Viterbi 算法找到最佳线路. 
```





### 摘要

从产品评论中挖掘观点的任务是提取产品实体并确定对实体的观点是正面的, 负面的还是中立的. 通过采用基于规则的统计方法或生成性学习模型 (例如隐马尔可夫模型 (HMM)), 已在此任务上实现了不错的性能. 在本文中, 我们提出了使用线性链条件随机场 (CRF) 进行判别的判断模型. CRF 可以自然地合并输入的任意, 非独立特征, 而无需在特征之间进行条件独立性假设. 这对于挖掘产品评论的观点尤其重要. 我们基于三个标准评估了我们的方法: 召回率, 准确性和提取实体的 F-score, 观点及其极性. 与欺他方法相比, 我们的方法被证明对完成观点挖掘任务更为有效. 



### 1 介绍

由于用户生成的评论包含来自用户体验的宝贵意见, 因此日益被认为在商业, 教育和电子商务中很有用. 例如, 在电子商务网站, 客户可以通过阅读其他客户对产品的评论来评估产品的质量, 这将有助于他们决定是否购买该产品. 如今, 许多电子商务网站, 例如 Amazon.com, Yahoo shopping, Epionions.com, 都允许用户自由发表意见. 在这些大型网站中, 评论的数量实际上已经达到了数千个, 因此, 对于潜在客户来说, 对所有这些网站进行审查都构成了挑战. 

为了解决该问题, 研究人员在 Web 意见挖掘上做了一些工作, 旨在从评论中发现必要的信息, 然后呈现给用户. 先前的工作主要采用基于规则的技术和统计方法. 最近, 提出了一种基于名为隐马尔可夫模型 (HMM) 的序列模型的新学习方法, 并被证明比以前的工作更有效. 但是, 基于 HMM 的方法仍然受到限制, 因为难以对输入字序列的任意依赖特征建模. 

为了解决这个限制, 在本文中, 我们特别研究了条件随机场 (CRF), 因为它是一种可区分的, 无向的图形模型, 可以潜在地建模重叠的相关特征. 在自然语言处理的先前工作中, CRF 被证明优于 HMM. 因此, 受早期发现的启发, 我们提出了一种基于 CRF 的线性链学习方法来挖掘和从 Web 上的产品评论中提取意见. 具体来说, 我们的目标是回答以下问题: (1) 如何定义特征函数来构建和限制我们的线性链 CRF 模型 ? (2) 如何从手动标记的数据中选择训练特定模型的标准 ? (3) 如何通过我们训练有素的模型自动提取产品实体并确定其相关的观点极性 ? 在实验中, 我们根据提取的实体和意见的三个评估指标对模型进行了评估: 召回率, 准确性和 F-score. 实验结果证明, 与相关的基于规则和基于 HMM 的方法相比, 我们提出的方法在完成网络意见挖掘任务方面具有更高的准确性. 

为了突出我们的南献, 我们已经证明吸, 基于 CRF 的线性链学习方法在整合语言特征以进行观点挖掘方面可以比 L-HMM 方法更好. 我们的工作中定义的特征函数对于模型构建已表明是键壮和有效的. 

因此, 本文的其余部分安排如下: 我们将在第 2 节中讨论相关工作, 并在第 3 节中详细介绍我们提出的基于 CRF 的意见挖掘学习方法. 在第 4 节, 我们将介绍实验设计和结果. 最后, 我们将结束工作并给出未来的方向. 





### 2 相关工作

迄今为止, 许多研究人员已尝试从用户评论 (或在某些文献中称为文档) 中提取观点值. 在文档级别, Turney 等人. 使用逐点互信息 (PMI) 来计算提取出的短语的平均语义方向 (SO), 以确定文档的极性. Pang 等研究了应用机器学习肢术解决电影评论数据的情感分类问题的有效性. Hatzivassilonglou 和 Wiebe 研究了动态形容词, 面向语义的形容词和可分级形容词对一个简单的主观分类器的影响, 并提出了一种可训练的方法, 该方法可统计地结合了两个等级能力指标. Wiebe 和 Riloff 提出了一个名为 OpinionFinder 的系统, 该系统可通过主观性分析自动识别文本中是否存在意见, 情感, 推测和其他隐私状态. Das 和 Chen 研究了财务文件的情感分类. 但是, 尽管上述作品都与情感分类有关, 但是它们只是使用情感来代表评论者的总体观点, 而没有找到评论者真正喜欢和不喜欢的特征. 例如, 对某个对象的整体否定情绪并不意味着 reviewe 不喜欢该对象的每个方面, 这实际上只能表明从该 reviewe 中总结出来的平均意见是消极的. 

为了深入了解评论者在文章中提到的几乎每个方面的观点, 一些研究人员试图在功能级别上挖掘和提取观点. Hu 和 Liu 提出了一种基于特征的意见汇总系统, 该系统通过在统计框架下使用关联规则来捕获频繁出现的特征词. 它提取客户表达了意见的产品的功能, 然后针对每个常见功能 (排名最高的功能之一) 给出一个评分, 而忽略不常见的功能. Popescu 和 Etzioni 删除了可能不是真实特征的常用名词短语, 从而改善了 Hu 和 Liu 的工作. 他们的方法可以识别部分关系并获得更好的精度, 但召回率却下聊很少. Scaffidi 等人提出了一种新的搜索系统, 称为 Red Opal, 该系统检查了先前的客户评论, 确定了产品功能, 然后对每个产品的每个产品功能进行评分. 当用户指定所需的产品功能时, Red Opal 使用这些分数来确定要退货的产品. 但是, 这些工作的局限性在于它们无法有效地识别出不常见的实体. 

与我们关注的焦点最相似的工作称为 OpinionMiner 的监督学习系统. 它是在词汇化的 HMM 框架下构建的, 该框架将多个重要的语言功能集成到一个自动学习过程中. 我们的方法与它的区别在于, 我们使用 CRF 来避免 HMM 固有的一些限制. 实际上, 它不能表示标签之间分布的隐藏状态和复杂的交互. 它即不能包含丰富的, 也不能包含重叠特征集. Miao 等人最近也尝试采用 CRF 来提取产品特征和观点, 但是他们没有使用 CRF 识别情感倾向, 也没有在不同方法之间进行比较. 



### 3 拟议方法

图 1 给出了我们方法的架构概述. 它可以分为四个主要步骤: (1) 预处理, 包括爬取原始评论数据和清理; (2) 标注评论数据 (3) 训练线性链 CRF 模型 (4) 应用模型到新的评论数据以获取观点. 



### 3.1 CRFs 学习模型

条件随机场 (CRF) 是无向图模型上的条件概率分布 [11]. 可以定义如下: 考虑图 $G=(V, E)$, 其中 $V$ 表示节点, $E$ 表示边缘. 令 $Y=(Y_{v})_{v \in V}$, $(X, Y)$ 是 $CRF$, 其中 $X$ 是已标注观测序列上的变量 (例如: 组成句子的一系列文字单词), $Y$ 是对应序列上的随机变量集合. $(X, Y)$ 遵守图表的 Markov 属性 (例如, 单词序列的部分语音标签). 形式上, 该模型定义了 $p(y | x)$, 以 $X$ 的全局观测为条件: 

$$\begin{aligned} p(y | x) = \frac{1}{Z(x)}{\prod_{i \in N}{\phi_{i}{(y_{i}, x_{i})}}} \quad (1) \end{aligned}$$ 

其中: $Z(x) = \sum_{y}{\prod_{i \in N}{\phi_{i}{(y_{i}, x_{i})}}}$ 是 $x$ 序列的所有状态上的规范化因子. 势通常在一组特征 $f_{k}$ 上分解, 如: 

$$\begin{aligned}  \phi_{i}(y_{i}, x_{i}) = \exp(\sum_{k}{\lambda_{k} f_{k}(y_{i}, x_{i})}) \quad (2) \end{aligned}$$ 

给定方程式 (1) 中定义的模型, 输入 $x$ 的最可能标记序列为: 

$$\begin{aligned} \hat{Y} = \underbrace{argmax}_{y} p(y | x) \quad (3) \end{aligned}$$ 



### 3.2 问题陈述

我们的目标是从评论中提取产品实体, 其中还包括意见极性. 根据 [1], 产品实体可以分为四类: 组件, 功能, 特性和意见. 请注意, 此处提到的功能是指产品的功能 (例如, 相机的尺寸, 重量), 与构造 CRF 模型的特征中的特征含义不同. 表 1 显示了实体的四个类别及其示例. 在我们的工作中, 我们遵循此分类方案. 

我们使用三种类型的标签来定义每个单词: 实体标签, 位置标签和意见标签. 我们使用产品实体的类别名称作为实体标签. 对于不是实体的单词, 我们使用字符 "B" 来表示. 通常, 一个实体可以是单个单词或短语. 对于短语实体, 我们为短语中的每个单词分配一个位置. 短语中的任何单词都有三个可能的位置: 短语的开头, 短语的中间和短语的结尾. 我们使用字符 "B", "M" 和 "E" 作为位置标记, 分别表示这三个位置. 对于 "意见" 实体, 我们进一步使用字符 "P" 和 "N" 分别表示肯定意见和否定意见极性, 并使用 "Exp" 和 "Imp" 分别表示明确意见和隐含意见. 在这里, 明确的意见意味着用户在评论中明确表达意见, 而隐含的意见意味着需要从评论中得出意见. 这些标签称为意见标签. 因此, 使用上面定义的所有标记, 我们可以标记任何单词及其在句子中的作用. 例如, 相机评论中的 "The image is good and its ease of use is satisfying" 的句子标记为: 

The(B) image(Feature-B) is(B) good(Opinion-B-P-Exp) and(B) its(B) ease(Feature-B) of(Feature-M) use(Feature-E) is(B) satisfying(Opinion-B-P-Exp). 

在这句话中, "image" 和 "ease of use" 都是相机的功能, "ease of use" 是一个短语, 因此我们添加 "-B", "-M" 和 "-E" 来指定位置短语中每个单词的数量. "Good" 是对功能 "image" 表达的显式的肯定意见, 因此其标签为 "Opinion-B-P-Exp" (这种标签组合在 [1] 中也称为混合标签). 不属于任何实体类别的其他单词都标记为 "B". 因此, 当我们获得每个单词的标签时, 我们可以获取它所引用的产品实体, 并确定其是否为 "Opinion" 实体. 通过这种方式, 可以将挖掘任务转换为自动标记任务. 然后可以将该问题形式化为: 给定一个单词序列 $W=w_{1}w_{2}w_{3}\cdots w_{n}$ 及其对应的词性 $S=s_{1}s_{2}s_{3}\cdots s_{n}$, 目标是找到一个合适的标签序列, 该标签序列可以根据等式 (3) 使条件似然最大化. 

$$\begin{aligned} \underbrace{argmax}_{T} p(T | W, S) = \underbrace{argmax}_{T} \prod_{i=1}^{N}{p(t_{i} | W, S, T^{(-i)})} \quad (4) \end{aligned}$$ 

在等式 (4) 中, $T^{(-i)} = \{ t_{1} t_{2} \cdots t_{i-1} t_{i+1} \cdots t_{N} \}$ (在我们的情况下是标签, 在一般概念中称为隐藏状态). 从这个等式, 我们可以看到位置 $i$ 处的单词标记取决于所有单词 $W = w_{1:N}$, 词性 $S=s_{1:N}$ 和标记. 不幸的是, 由于该方程涉及太多参数, 因此很难进行计算. 为了降低复杂度, 我们采用线性链 CRF 作为近似值来限制标签之间的关系的. 它是如图 2 所示的图形结构 (在图中, $Y$ 构成简单的一阶链). 在线性链 CRF 中, 图中的所有节点都形成一条线性链, 每个要素仅涉及两个连续的隐藏状态. 等式 (4) 因此可以重写为: 

$$\begin{aligned} \underbrace{argmax}_{T} p(T | W, S) = \underbrace{argmax}_{T} \prod_{i=1}^{N}{p(t_{i} | W, S, t_{i-1})} \quad (5) \end{aligned}$$ 



### 3.3 特征函数

从上面的模型中, 我们可以看到仍有许多参数需要处理. 为了使模型更具有可计算性, 我们需要定义观察状态 $W=w_{1:N}$, $S=s_{1:N}$ 和隐藏状态 $T=t_{1:N}$ 之间的关系, 以减少不必要的计算. 因此, 特征函数作为 CRF 的重要构造, 对于解决我们的问题至关重要. 假设 $w_{1:N}$, $s_{1:N}$ 为观察值 (即单词的序列及其对应的词性部分), $t_{1:N}$ 为隐藏标签 (即标签). 在我们的线性链 CRF 的情况下, 特征函数的一般形式为 $f_{i}(t_{j-1}, t_{j}, w_{1:N}, s_{1:N}, n)$, 它看起来是一对相邻的状态 $t_{j-1}, t_{j}$, 整个输入序列 $w_{1:N} $ 以及 $s_{1:N}$ 和当前单词的位置. 例如, 我们可以定义一个简单的生成二进制值的特征函数: 如果当前单词 $w_{j}$ 为 "good", 对应的词性 $s_{j}$ 为 "JJ" (表示单个形容词), 当前状态 $t_{j}$ 为 "opinion",  则返回值为 1. 

$$f_{i}(t_{j-1}, t_{j}, w_{1:N}, s_{1:N}, j) = \left\{ \begin{aligned} & 1 \space if \space w_{j} = good, s_{j} = JJ \space and \space t_{j} = Opinion \\ & 0 \space otherwise \end{aligned} \right. \quad (6)$$ 

将特征函数与方程式 (1) 和方程式 (2) 结合, 我们得到: 

$$\begin{aligned} p(t_{1:N} | w_{1:N}, s_{1:N}) = \frac{1}{Z} \exp{(\sum_{j=1}^{N} \sum_{i=1}^{F} {\lambda_{i} f_{i} (t_{j-1}, t_{j}, w_{1:N}, s_{1:N}, j)})} \quad (7) \end{aligned}$$ 

根据等式 (7), 特征函数 $f_{i}$ 取决于其相应的权重 $\lambda_{i}$. 也就是说, 如果 $\lambda_{i} > 0$, 且 $f_{i}$ 有效 (即 $f_{i} = 1$), 则将增加标签序列 $t_{1:N}$ 的概率; 如果 $\lambda_{i} < 0$, 且 $f_{i}$ 无效 ($f_{i} = 0$), 则它会增加将降低标签序列 $t_{1:N}$ 的可能性. 

特征函数的另一个示例可以是: 

$$f_{i}(t_{j-1}, t_{j}, w_{1:N}, s_{1:N}, j) = \left\{ \begin{aligned} & 1 \space if \space w_{j} = good, s_{j+1} = NN \space and \space t_{j} = Opinion \\ & 0 \space otherwise \end{aligned} \right. \quad (8)$$ 

在这种情况下, 如果当前词是 "good" 例如是在短语 "good image" 中, 则等式 (6) 和 (8) 中的特征函数将同时被激活. 这是一个交叉特征的示例, 这在 HMMs 中不能被处理. 实际上, HMMs 不能考虑下一个单词, 也不能使用交叉的特征. 

除了使用线性链 CRF 简化隐藏状态 $T$ 中的关系外, 我们还定义了几种不同类型的特征函数来指定 $W$, $S$, 和 $T$ 之间的状态转换结构. 对于不同类别的特征, 不同的状态转换特征基于不同的 Markov 顺序.  在这里, 我们定义一阶特征: 

1. 当前标签 $t_{j}$ 的分配应该仅取决于当前单词. 特征函数表示为 $f(t_{j}, w_{j})$. 
2. 当前标签 $t_{j}$ 的分配应该仅取决于当前词性. 特征函数表示为 $f(t_{j}, s_{j})$. 
3. 当前标签 $t_{j}$ 的分配应该同时取决于当前单词和词性. 特征函数表示为 $f(t_{j}, s_{j}, w_{j})$. 

三种类型的特征函数都是一阶的, 仅在当前状态下检查输入. 我们还定义了在当前状态和先前状态的上下文中检查的一阶+过渡特征和二阶特征. 我们没有定义三阶或更高阶特征, 因为它们会造成数据稀疏问题, 并且在训练过程中需要更多内存. 表 2 显示了我们在模型中定义的所有特征函数. 



### 3.4 CRFs 训练

定义图形和特征函数后, 将固定模型. 训练的目的是识别所有 $\lambda_{1:N}$ 的值. 通常可以根据领域知识设置 $\lambda_{1:N}$. 但是, 在我们的情况下, 我们从训练数据中学习了 $\lambda_{1:N} $. 完全标记的评论数据为 $\{ (w^{(1)}, s^{(1)}, t^{(1)}, \cdots , (w^{(M)}, s^{(M)}, t^{(M)})) \}$, 其中 $w^{(i)} = w_{1:N_{i}}^{(i)}$ (第 $i$ 个词序列), $s^{(i)} = s_{1:N_{i}}^{(i)}$ (第 $i$ 个词性序列), $t^{(i)} = t_{1:N_{i}}^{(i)}$ (第 $i$ 个标记序列). 鉴于在 CRF 中, 我们定义了条件概率 $p(t | w, s)$, 参数学习的目的是基于训练数据最大化条件似然: 

$$\begin{aligned} \sum_{j=1}^{M}{\log {p(\mathbf{t}^{(i)} | \mathbf{w}^{(i)}, \mathbf{s}^{(i)})}} \quad (9) \end{aligned}$$ 

为了避免过度拟合, 对数似然率通常会因参数的某些先验分布而受到惩罚. 常用的分布是零均值高斯分布. 如果 $\lambda \sim N(0, \sigma^{2})$, 则等式 (9) 变为: 

$$\begin{aligned} \sum_{j=1}^{M}{\log{p(t^{(i)} | w^{(i)}, s^{(i)})} - \sum_{i}^{F}{\frac{\lambda_{i}^{2}}{2 \sigma^{2}}}} \quad (10) \end{aligned}$$ 



该方程是凹的, 因此 $\lambda$ 具有一组唯一的全局最优值. 我们通过计算目标函数的梯度来学习参数, 并在称为受限内存 BFGS (L-BFGS) 的优化算法中使用梯度. 

目标函数的梯度正式计算如下: 

$$\begin{aligned} & \frac{\partial}{\partial \lambda_{k}}{\sum_{j=1}^{m}{\log{p(t^{(j)} | w^{(j)}, s^{(j)})}} - \sum_{i}^{F}{\frac{\lambda_{i}^{2}}{2 \sigma^{2}}}} \\ =& \frac{\partial}{\partial \lambda_{k}}{\sum_{j=1}^{m}{(\sum_{n} \sum_{i} \lambda_{i} f_{i}(t_{n-1}, t_{n}, w_{1:N}, s_{1:N}, n) - \log{T^{(j)}} )}} - \sum_{i}^{F}{\frac{\lambda_{i}^{2}}{2 \sigma^{2}}} \\ =& \sum_{j=1}^{m} \sum_{n} {f_{k}(t_{n-1}, t_{n}, w_{1:N}, s_{1:N}, n)} - \sum_{j-1}^{m} \sum_{n} {E_{t_{n-1}^{'}, t_{n}^{'}}[f_{k}(t_{n-1}^{'}, t_{n}^{'}, w_{1:N}, s_{1:N}, n)] - \frac{\lambda_{k}}{\sigma^{2}}}  \end{aligned} \quad (11) $$ 

在等式 (11) 中, 第一项是训练数据中特征 $i$ 的经验计数, 第二项是在当前训练模型下该特征的预期计数, 而第三项是通过先验分布生成的. 因此, 导数测量的是当前模型下特征的经验计数与预期计数之间的差异. 假设在训练数据中特征 $f_{k}$ 出现 $A$ 次, 而在当前模型下, $f_{k}$ 的期望计数为 $B$: 当 $|A| = |B|$, 则导数为 0. 因此, 训练过程是找到与两个计数匹配的 $\lambda $. 







































