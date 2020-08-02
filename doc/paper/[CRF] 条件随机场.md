## 条件随机场

http://pages.cs.wisc.edu/~jerryzhu/cs838/CRF.pdf



### 信息提取

当前的 NLP 技术无法完全理解一般的自然语言文章. 但是, 它们仍然可以用于受限任务. 一个示例是信息提取. 例如, 可能要从研究人员的网页中提取标题, 作者, 年份和会议名称. 或者一个人想要从新闻文章 (NER, 称为实体识别) 中识别人员, 位置, 组织名称. 这些对将 Web 上的自由文本自动转换成知识数据库很有用, 并构成许多 Web 服务的基础. 

基本的信息提取技术是将问题视为文本序列标记问题. 例如, 标签集可以是 {标题, 作者, 年份, 会议, 其他} 或 {人物, 位置, 组织, 其他}. 因此, HMM 已自然而然地成功地应用于信息提取. 但是, HMM 难以建模输出的重叠, 非独立特征. 例如, HMM 可以通过 $p(x | z)$ 指定哪些单词可能出现在给定状态 (tag) 中. 但是通常单词的词性以及周围单词的词性, n-gram 字符, 大写形式都带有重要信息. HMM 无法轻松地为它们建模, 因为生成的故事限制了状态变量可以生成的内容. 条件随机场 (CRF) 可以对这些重叠的非独立特征进行建模. 线性链 CRF 是一种特殊情况, 可以认为是 HMM 的无向图模型版本. 它与 HMM 一样有效, 在 HMM 中, sumproduct 算法和 max-product 算法仍然适用. 



### 2 CRF 模型

令 $x_{1:N}$ 为观察值 (例如文档中的单词), $z_{1:N}$ 为隐藏标签 (例如标签). 线性链条件随机场定义条件概率 (而 HMM 定义关节). 

$$\begin{aligned} p(z_{1:N} | x_{1:N}) = \frac{1}{Z} \exp(\sum_{n=1}^{N} \sum_{i=1}^{F} \lambda_{i} f_{i} (z_{n-1}, z_{n}, x_{1:N}, n)) \quad (1) \end{aligned}$$ 

让我们详细遍历该模型. 标量 $Z$ 是归一化因子或分区函数, 以使其成为有效概率. $Z$ 定义为序列的指数数目之和. 

$$\begin{aligned} Z = \sum_{z_{1:N}} \exp(\sum_{n=1}^{N} \sum_{i=1}^{F} \lambda_{i} f_{i} (z_{n-1}, z_{n}, x_{1:N}, n)) \quad (2) \end{aligned}$$ 

因此通常很难计算. 注意 $Z$ 隐式取决于 $x_{1:N}$ 和参数 $\lambda$. 

大的 $\exp()$ 函数是有历史原因的, 它与指数族分布有关. 现在, 仅需注意和 $f()$ 可以采用任间实数值, 并且整个 $\exp$ 函数将为非负数. 

在 $\exp()$ 函数中, 我们求和 $n=1, \cdots , N$ 个单词在序列中的位置. 对于每个位置, 我们求和等于 $i=1, \cdots , F$ 加权特征. 标量 $\lambda_{i}$ 是特征函数 $f_{i}()$ 的权重. $\lambda_{i}$ 是 CRF 模型的参数, 需要通过学习得到, 类似于 HMMs 中的 $\theta = \{ \pi, \phi, A \}$. 



### 3 特征函数

特征函数是 CRF 的关键组件. 在线性链 CRF 的特殊情况下, 特征函数的一般形式是 $f_{i}(z_{n-1}. z_{n}, x_{1:N}, n)$, 它查看了一对相邻的状态 $z_{n-1}, z_{n}$, 整个输入序列 $x_{1:N}$, 以及我们在序列 $n$ 中的位置. 这些是产生实数值的任意函数. 

例如, 我们可以定义一个简单的生成二进制值的特征函数: 如果当前单词是 John, 并且当前状态 $z_{n}$ 是 PERSON, 则它为 1: 

$$\begin{aligned} f_{1}(z_{n-1}, z_{n}, x_{1:N}, n) = \left\{ \begin{aligned} & 1 \space \text{if }\space z_{n} = PERSON \space \text{and} \space x_{n} = John \\ & 0 \space \text{otherwise} \end{aligned} \right. \end{aligned} \quad (3) $$ 

如何使用此特征. 这取决于其相应的权重 $\lambda_{1}$. 如果 $\lambda_{1} > 0$, 则无论何时 $f_{1}$ 处于活跃状态 (即我们在句子中看到单词 John 并为其分配标签 PERSON), 它都会增加标签序列 $z_{1:N}$ 的可能性. 这是说 CRF 模型应该使用标签 PERSON 代替 John 单词的另一种说法. 另一方面, 如果 $\lambda_{1} < 0$, 则 CRF 模型将尝试避免使用标签 PERSON 给 John. 哪种方法正确 ? 一个人可以通过领域知识设置 $\lambda_{1}$ (我们知道它可能应该是肯定的), 或者从语料库中学习 $\lambda_{1}$ (让数据告诉我们), 或者两者 (将领域知识视为 $\lambda_{1} $ 先验知识). 注意 $\lambda_{1}$, $f_{1}$ 一起, 等价于 HMM 的 $\phi$ 参数的 $\log$ 对数. $p(x = \text{John} | z = \text{PERSON})$. 

 再举一个例子, 考虑: 

$$\begin{aligned} f_{2}(z_{n-1}, z_{n}, x_{1:N}, n) = \left\{ \begin{aligned} & 1 \space \text{if }\space z_{n} = PERSON \space \text{and} \space x_{n+1} = said \\ & 0 \space \text{otherwise} \end{aligned} \right. \end{aligned} \quad (4) $$ 

如果当前标签是 PERSON, 下一个单词是 "said", 岀此特征激活. 因此, 人们希望该特征正数 $\lambda_{2}$. 此外, 在如句子 "John said so. " 中, $f_{1}$, $f_{2}$ 特征函数都会被激活. $z_{1} = PERSON$. 这是一个重叠特征的示例. 它将 $z_{1} = PERSON$ 的信念提高到 $\lambda_{1} + \lambda_{2}$. HMM 无法做到这一点: HMM 无法查看下一个单词, 也不能使用重叠特征. 

下一个特征示例类似于 HMM 中的转换矩阵 A. 我们可以定义: 

$$\begin{aligned} f_{3}(z_{n-1}, z_{n}, x_{1:N}, n) = \left\{ \begin{aligned} & 1 \space \text{if }\space z_{n-1} = OTHER \space \text{and} \space z_{n} = PERSON \\ & 0 \space \text{otherwise} \end{aligned} \right. \end{aligned} \quad (5) $$ 

如果我们看到特定的标签转换 (OTHER, PERSON), 则此特征激活. 请注意, $\lambda_{3}$ 的值指定了 HMM 表示法中从 OTHER 到 PERSON 或 $A_{OTHER, PERSON}$ 的 $\log $ 对数转换概率的等价形式. 以类似于方式, 我们可以定义所有 $K^{2}$ 过渡特征, 其中 $K$ 是标签集的大小. 

当然, 特征不限于二进制函数. 允许使用任何实值函数. 





### 4 无向图模型 (Markov 随机场)

CRF 是无向图形模型的特殊情况, 也称为马尔可夫随机场. 团是图中完全连接的节点的子集 (在任何两个节点之间都有一条边). 最大团不是任何其他团的子集. 令 $X_{c}$ 是最大团 $c$ 涉及的节点集. 令 $\psi(X_{c})$ 为任意的非负实值函数, 称为势函数. 特别是 $\psi(X_{c})$ 不需要标准化. 马尔可夫随机场将节点状态上的概率分布定义为图中所有团的潜在函数的归一化乘积. 

$$\begin{aligned} p(X) = \frac{1}{Z} \prod_{c}{\psi(X_{c})} \quad (6) \end{aligned}$$ 

其中 $Z$ 是正则化因子. 在线性链 CRF 的特殊情况下, 团对于一对状态 $z_{n-1}$, $z_{n}$ 以及对应的 $x$ 个节点, 其中: 

$$\begin{aligned} \psi = \exp(\lambda f) \quad (7) \end{aligned}$$ 

实际上, 这确定也是与因子图表示的直接联系. 每个团可以由一个具有因子 $\psi(X_{c})$ 的因子节点表示, 该因子节点连接到 $X_{c}$ 中的每个节点, 有一个表示 $Z$ 的加法特殊因子节点. 

可喜的结果是, 求和积算法和最大和算法立即适用于马尔可夫随机场 (尤其是 CRF). 消息传递期间可以忽略与 $Z$ 对应的因数. 



### 5 CRF 训练

训练涉及查找参数 $\lambda $. 为此, 我们需要完全标记的数据序列 $\{ (\mathbf{x}^{(1)}, \mathbf{z}^{(1)}), \cdots , (\mathbf{x}^{(m)}, \mathbf{x}^{(m)}) \}$, 其中 $\mathbf{X}^{(1)} = x_{1:N_{1}}^{(1)}$ 表示第 1 个观测序列, 依此类推. 由于 CRF 定义了条件概率 $p(z | x)$, 因此参数学习的适当目标是使训练数据的条件可能性最大. 

$$\begin{aligned} \sum_{j=1}^{m}{\log{p(\mathbf{z}^{(j)} | \mathbf{x}^{(j)})}} \quad (8) \end{aligned}$$ 

通常, 你也可以将高斯先验应用到 $\lambda$ 以规范化训练 (即平滑). 如果 $N(0, \sigma^{2})$, 则目标变为: 

$$\begin{aligned} \sum_{j=1}^{m}{\log{p(\mathbf{z}^{(j)} | \mathbf{x}^{(j)})} - \sum_{i}^{F}{\frac{\lambda_{i}^{2}}{2 \sigma^{2}}}} \quad (9) \end{aligned}$$ 

好消息是, 目标是凹函数, 因此它具有一组独特的最佳值. 坏消息是封闭式解决方案. 

标准参数学习方法是计算目标函数的梯度, 并在诸如 L-BFGS 的优化算法中使用梯度. 目标函数的梯度计算如下: 

$$\begin{aligned} & \frac{\partial}{\partial \lambda_{k}}{\sum_{j=1}^{m}{\log{p(\mathbf{z}^{(j)} | \mathbf{x}^{(j)})}} - \sum_{i}^{F}{\frac{\lambda_{i}^{2}}{2 \sigma^{2}}}} \\ =& \frac{\partial}{\partial \lambda_{k}}{\sum_{j=1}^{m}{(\sum_{n} \sum_{i} \lambda_{i} f_{i}(z_{n-1}^{(j)}, z_{n}^{(j)}, \mathbf{x}^{(j)}, n) - \log{Z^{(j)}} )}} - \sum_{i}^{F}{\frac{\lambda_{i}^{2}}{2 \sigma^{2}}} \\ =& \sum_{j=1}^{m} \sum_{n} {f_{k}(z_{n-1}^{(j)}, z_{n}^{(j)}, \mathbf{x}^{(j)}, n)} - \sum_{j-1}^{m} \sum_{n} {E_{z_{n-1}^{'}, z_{n}^{'}}[f_{k}(z_{n-1}^{'}, z_{n}^{'}, \mathbf{x}^{(j)}, n)] - \frac{\lambda_{k}}{\sigma^{2}}}  \end{aligned} \quad (12) $$ 

其中我们使用事实: 

$$\begin{aligned} \frac{\partial}{\partial \lambda_{k}} \log{Z} &= E_{z'}[\sum_{n}{f_{k}(z_{n-1}^{'}, z_{n}^{'}, \mathbf{x}, n)}] \quad &(13) \\ &= \sum_{n}{E_{z_{n-1}^{'}, z_{n}^{'}}[f_{k}(z_{n-1}^{'}, z_{n}^{'}, \mathbf{x}, n)]} \quad &(14) \\ &= \sum_{n} \sum_{z_{n-1}^{'}, z_{n}^{'}} {p(z_{n-1}^{'}, z_{n}^{'} | \mathbf{x}) f_{k}(z_{n-1}^{'}, z_{n}^{'}, \mathbf{x}, n)} \quad &(15) \end{aligned}$$ 

请注意, 边缘边标概率 $p(z_{n-1}^{'}, z_{n}^{'} | \mathbf{x})$ 在当前参数下, 这正是和积算法可以计算的. (12) 中的偏导数具有直观的解释. 让我们忽略前面的术语 $\lambda_{k} / \sigma^{2}$. 导数的形式为 (特征 $f_{k}$ 的观察计数) 减云 (特征 $f_{k}$ 的预期计数). 当两者相同时, 导数为零, 并且不再有改变 $\lambda_{k}$ 的动机. 因此, 我们认为训练可以被认为是找到与两个指标匹配的 $\lambda $. 



### 6 特征选择

NLP 中的一种常见做法是定义大量候选特征, 并让数据选择一个小的子集, 以在称为特征选择的过程中在最终 CRF 模型中使. 通常会在两个阶段中提出候选特征: 

1. 原子候选特征. 这些通常是针对单词和标签的特定组合的简单测试, 例如 (x=John, z=PERSON), (x=John, z=LOCATION), (x=John, z=ORGANIZATION) 等. 有 $V K$ 这样的 "单词-身份" 候选特征, 显然是很多. 尽管这被称为"单词-身份"测试, 但应将其理解为与每个标签值结合使用. 类似地, 可以测试单词是否为大写字母, 相邻单词的身份, 单词的词性等. 状态转换特征也是原子的. 从大量的原子候选特征中, 通过改善 CRF 模型的程度 (例如, 训练集可能性的增加) 选择少量特征. 
2. "Grow" 候选特征. 组合特征以形成更复杂的特征是很自然的. 例如, 一个人可以测试当前单词是否大写, 下一个单词是 "Inc.", 两个标签都为 ORGANIZATION. 但是, 复杂特征的数量呈指数增长. 一种折衷的方案是, 到目前为止, 仅通过在所选要素上添加候选原子, 或通过添加一个原子加法或其他简单的布尔运算来扩展候选特征. 通常, 将任何剩余的原子候选特征添加到增长集中. 选择了少量特征, 并将其添加到现有特征集中. 重复此阶段, 直到添加足够的特征. 





































