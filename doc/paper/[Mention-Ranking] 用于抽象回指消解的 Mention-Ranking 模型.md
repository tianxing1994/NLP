## 用于抽象回指消解的 Mention-Ranking 模型

https://arxiv.org/pdf/1706.02256.pdf



### 摘要

解决抽象回指是文本理解的重要但困难的任务. 然而, 随着表示学习的最新进展, 此任务已成为更切实的目标. 抽象照应的一个主要特性是它在照应句中嵌入的照应与其 (通常是名词性的) 先行词之间建立了联系. 我们提出了一种 Mention-Ranking 模型, 该模型通过 LSTM-Siamese Net 学习抽象照应如何与其先行词相关. 通过生成人为的反指句-先行对, 我们克服了训练数据的不足. 在壳名词解析方面, 我们的模型优于最新结果. 我们还报告了 ARRAU 语料库的抽象照应子集的第一个基准测试结果. 由于标称和代词照应的混合以及更大范围的混杂因素, 该语料库提出了更大的挑战. 我们发现模型变体在不对个别照应数据进行训练的情况下优于名义照应的基准, 但对于代词照应仍然落后. 我们的模型选择语法上合理的候选者, 并且-如果不考虑语法, 则使用更深层的特征来区分候选者. 



### 1 介绍

当前回指 (或共指) 消解的研究集中于消解指代现实世界中具体对象或实体的名词短语, 这可以说是最常见的类型. 与这些不同的是抽象照应 (AA) (Asher, 1993) 的各种类型, 其中提到了命题, 事实, 事件或性质. 下面的 (1) 中给出了一个例子. 虽然最近的方法解决了选定的抽象壳名词的解析问题 (Kolhatkar 和 Hirst, 2014), 但我们的目标是解决各种各样的抽象照应, 例如 (1) 中的 NP 这种趋势以及代词照应 (this, that, or). 此后, 我们将包含抽象照应的句子称为照应句 (AnaphS), 并将照应所指的成分称为先行词 (Antec). 

```text
(1) Ever-more powerful desktop computers, designed with one or more microprocessors as their ”brains”, are expected to increasingly take on functions carried out by more expensive minicomputers and mainframes. ”[Antec The guys that make traditional hardware are really being obsoleted by microprocessor-based machines]”, said Mr. Benton. [AnaphS As a result of this trendAA, longtime powerhouses HP, IBM and Digital Equipment Corp. are scrambling to counterattack with microprocessor-based systems of their own.]
```

解决些任务的主要障碍是缺少足够数量的带注释的训练数据. 我们提出了一种生成大量训练实例的方法, 这些实例涵盖了广泛的抽象照应类型. 这使我们能够使用在相关任务中显示出巨大成功的神经方法: 共指消解, 本文蕴涵, 学习文本相似性和话语关系感分类. 我们的模型的灵感来自于共指消解的 Mention-Ranking 模型, 并将其与 Siamese Net 相结合. 给定一个照应性句子和一个候选先行词, LSTM-Siamese Net 学习候选词和该照应词的表示形式共享空间中的句子. 这些表示被组合为联合表示, 用于计算表征它们之间关系的分数. 学习的分数用于选择给定照应句子的最高得分先行候选者, 从而选择其照应. 我们一次考虑一个照应词, 并在输入中提供照应词的上下文嵌入和隐喻词首的嵌入, 以表征每个单独的照应词-与 Zhou 和 Xu (2015) 提出的用于个体化的编码相似, 在 SRL 中乘以出现的谓词. 通过更深入的检查, 我们表明该模型学习了照应句中的照应词与其先行词之间的关系. 图 1 显示了我们的架构. 

与其他工作相比, 我们生成训练数据的方法并不局限于特定类型的回指, 例如 shell 名词或回指连词. 它产生大量实例, 并且很容易适应其他语言. 这使我们能够为抽象回指解析建立一个健壮的, 知识丰富的模型, 该模型可以轻松扩展到多种语言. 

我们在 Kolhatkar 等人的 shell 名词解析数据集上评估模型. 并证明它优于他们最新的结果. 此外, 我们报告了 ARRAU 语料库中不受限制的抽象回指实例的模型结果. 据我们所知, 这提供了有关该数据子集的第一个最新基准. 

我们的 TensorFlow 模型和脚本提取数据的实现可以在以下位置找到: https://github.com/amarasovic/neural-abstract-anaphora



### 2 相关的先前工作

**抽象照应**已在语言学中进行广泛的研究, 并显示出在语义先行类型, 抽象程度和一般话语属性方面的特定属性. 与名词照应相比, 抽象照应很难消解, 因为协议和词汇匹配特征不适用. 对于人类来说, 抽象回指的注释也很困难, 因此, 仅构建了一些较小规模的语料库. 我们对 ARRAU 语料库的子集进行评估, 该子集包含抽象照应和 Kolhatkar 等人使用的 shell 名词语料库. 我们不知道其他可免费获得的抽象照应数据集. 

**自动消解抽象回指**的工作很少. 早期的工作专注于口头语言, 它具有特定的特性. 最近, 事件共指已使用基于特征的分类器解决. 事件共指仅限于事件的子类, 通常着重于动词 (短语) 和名词 (短语) 之间的共指, 它们具有相似的抽象水平 (例如, 购买-获取), 而特别关注 (代词) 指代指称. 抽象回指通常包含一个完整的子句先行词, 该词被高度抽象的 (代词) 名词回指所指, 如 (1) 所示. 

Rajagopal 等. 提出了一个解决生物医学文本中涉及单个或多个条款的事件的模型. 但是, 与其为给定事件选择正确的先行子句 (我们的任务), 不如将其模型限制为将事件分为六个抽象类别: changes, responses, analysis, context, finding, observation, 基于其周围的上下文. 尽管相关, 但它们的任务与成熟的抽象回指解决方案任务不具有可比性, 因为已知要分类的事件是相互关联的, 并且是从一组受限制的抽象类型中选择的. 

与我们的工作更相关的是 Anand 和 Hardt, 他们提出了基于小型训练数据集的使用经典机器学习进行水闸的先行排名帐户. 他们采用建模距离, 包容性, 话语结构以及内容和词汇相关性 (效果较差) 的特征. 与我们的工作最接近的是 Kolhatkar 等. 和 Kolhatkar 和 Hirst 使用经典的机器学习技术对 shell 名词进行解析. shell 名词是抽象名词, 例如事实, 可能性或问题, 只能与它们的 shell 内容 (如 (2) 中嵌入子句或如 (3) 中的前子句) 一起解释. KZH13 将先行词在先验话语中出现的 shell 名词称为回指 shell 名词 (ASN), 而将回指 shell 名词 (CSN) 称为隐喻 shell 名词. 

```text
(2) Congress has focused almost solely on the fact that [special education is expensive - and that it takes away money from regular education.]
```

```text
(3) Environmental Defense [...] notes that [Antec Mowing the lawn with a gas mower produces as much pollution [...] as driving a car 172 miles.] [AnaphS This fact may [...] explain the recent surge in the sales of [...] old-fashioned push mowers [...]].
```

KZH13 提出了一种解决六个典型 shell 名词的方法, 原因是观察到 CSN 仅基于它们的语法结构就很容易解析, 并且假设 ASN 与它们的嵌入式 (CSN) 对应物具有语言特性. 他们手动制定了规则以识别 CSN 的嵌入子句 (即, 前奏的前奏), 并在此类情况下训练了 SVMrank. 然后, 将训练好的 SVMrank 模型用于解析 ASN. KH14 推广了他们的方法, 以便能够为任何给定的 shel 名词创建训练数据, 但是, 他们的方法大量利用了 shell 名词的特定属性, 不适用于其他类型的抽象照应. 

Stede 和 Grishina 研究了德国人的一个相关现象. 他们研究了固有的隐喻连接词 (例如 demzufolge - 据此), 可用于在即时上下文中访问其抽象先行词. 但是, 此类连接词的类型受到限制, 并且研究表明, 此类连接词通常与名词照应模棱两可, 并且需要消除歧义. 我们得出的结论是, 不能轻易使用它们自动获取先行词. 

在我们的工作中, 我们探索了一个不同的方向: 我们使和识别嵌入句子成分的一般模式来构造人工训练数据, 这使我们能够为抽象照应提取相对安全的训练数据, 以捕获广泛的照应-先行词关系, 并将其应用于训练模型以解决不受约束的抽象照应. 

实体共指消解的最新工作提出了强大的基于神经网络的模型, 我们将应用于抽象回指解析的任务. 与我们的任务最相关的是 Clark 和 Manning 提出的 Mention-Ranking 神经共指模型, 以及 Clark 和 Manning 提出的改进模型, 该模型整合了损失函数, 该模型学习了不同的知识. 特征表示, 用于隐喻检测和先行词排名. 

Siamese Nets 通过优化表示所导致的度量损失来区分相似和不相似的样本对. 它被广泛用于视觉, 以及在 NLP 中用于语义相似性, 含竟性, 查询规范化和 QA. 





### 3 Mention-Ranking 模型

给定带有明显照应 (mention) 和候选先行词 $c$ 的照应句子 $s$, Mention-Ranking (MR) 模型使用 LSTM-Siamese Net 生成的表示法为 $pair(c, s)$ 分配分数. 得分最高的候选词被分配给回指句子中的标记回指. 图 1 显示了模型. 我们使用双向长短期记忆来学习回指句子 $s$ 和候选先行词 $c$ 的表示. 一个 bi-LSTM 应用于照应词 $s$ 和候选先行词 $c$, 因此称为 siamese. 每个单词都由一个向量 $w_{i}$ 表示, 该向量通过以下方式构建: 将单词的嵌入词串联, 回指的上下文 (回指词组, 前一个词和下一个词的嵌入词的平均), 回指词组的头部, 最后, 候选单词构成标记的嵌入, 如果单词在回指句子中, 则为 $S$ 构成标记. 对于每个序列 $s$ 或 $c$, 将单词向量 $w_{i}$ 依次馈入 bi-LSTM, 后者从前向通过产生输出, $\overrightarrow{h}_{i}$, 后向输出 $\overleftarrow{h}_{i}$. 第 $i$ 个单词的最终输出定义为 $h_{i} = [\overrightarrow{h}_{i}; \overleftarrow{h}_{i}]$. 为了获得完整序列 $h_{s}$ 或 $h_{c}$ 的表示形式, 除与填充 token 相对应的所有输出外, 所有输出均取平均值. 

为避免忘记序列的组成标签, 我们将嵌入的相应标签与 $h_{s}$ 或 $h_{c}$ 连接 (我们称其为标签信息的快捷方式). 所得向量被馈送到指数线性单元 (ELUs) 的前馈层中, 以产生序列的最终表示 $\widetilde{h}_{s}$ 或 $\widetilde{h}_{c}$. 

从 $\widetilde{h}_{s}$ 和 $\widetilde{h}_{c}$ 我们计算向量 $h_{c,s} = [| \widetilde{h}_{c} - \widetilde{h}_{s}|; \widetilde{h}_{c} \odot \widetilde{h}_{s}]$, 其中 $|-|$ 表示逐元素减法的绝对值, $\odot$ 表示逐元素相乘. 然后 $h_{c,s}$ 被导入ELUs 的前向层以获得最后的表示 $\widetilde{h}_{c,s}$, 作为 $pair(c, s)$ 的表示, 最后, 通过对联合表示应用单个完全连接的线性层, 计算代表它们之间相关性的 $pair(c, s)$ 的分数. 

$$\begin{aligned} score(c, s) = W\widetilde{h}_{c,s} + b \in \mathbb{R} \quad (1) \end{aligned}$$ 

其中 $\mathbf{W}$ 是一个 $1 \times d$ 的权重矩阵, 向量 $\widetilde{h}_{c,s}$ 的维度为 $d$. 我们使用 Wiseman 等人的最大收益率训练目标来训练描述的 Mention-Ranking 模型. 用于之前的排名子任务. 假设训练集: $D = \{ (a_{i}, s_{i}, T(a_{i}), N(a_{i})) \}_{i=1}^{n}$, 其中 $a_{i}$ 是第 $i$ 个抽象回指, $s_{i}$ 是对应的回指句子, $T(a_{i})$ 是 $a_{i}$ 的先行词, $N(a_{i})$ 为 $a_{i}$ 的先行, 但不是它的真实先行词的集合 (负候选). 令 $\widetilde{t}_{i} = \text{argmax}_{t \in T(a_{i})} score(t_{i}, s_{i})$ 为$a_{i}$ 的最高得分的先行词. 损失函数如下: 

$$\begin{aligned} \sum_{i=1}^{n}{\max{(0, \underbrace{\max}_{c \in N(a_{i})} \{1 + score(c, s_{i}) - score(\widetilde{t}_{i}, s_{i})\})}} \end{aligned}$$ 



### 4 训练数据构造

我们通过利用包含一个常见结构来创建用于抽象回指解析的大规模训练数据, 该结构由带有嵌入句子 (补语或副词) 的动词组成 (参见图 2). 我们在经过分析的语料库中检测到这种模式, "cut off" $S'$ 成分, 并用适当的照应替换它以创建照应句 (AnaphS), 而 $S$ 产生先行词 (Antec). 由于承载动词的子句和嵌入的句子之间存在着多种多样的语义或话语关系, 因此该方法涵盖了多种回指. 首先, 模式适用于嵌入句子参数的动词. 在 (4) 中, 动词疑问词在嵌入句子及其句子补语之间建立了特定的语义关系. 

```text
(4) He doubts [S0 [S a Bismarckian super state will emerge that would dominate Europe], but warns of ”a risk of profound change in the [..] European Community from a Germany that is too strong, even if democratic”].
```

从中我们提取出一个将在欧洲占主导地位的人为先行的 Bismarckian 超级国家及其相应的回指句. 他对此表示怀疑, 但是警告说 "a risk of profound change ... even if democratic", 一组预定义的适当照应中的一个 (这里: this, that, it),. 与 (4) 相反, 当 $s'$ 开头被明显的补语 (表示怀疑) 填充时, 使用表 1 中的第二行. 表 1 中的其余行适用于不同类型的状语从句. 

状语从句用嵌入句子编码特定的语篇关系, 通常用它们的连词表示. 例如, 在 (5) 中, 因果关系与原因 (嵌入句子) 及其影响 (嵌入句子) 有关: 

```text
(5) There is speculation that property casualty firms will sell even more munis [S0 as [S they scramble to raise cash to pay claims related to Hurricane Hugo [..] ]].
```

我们随机替换因果连词, 因为与适当调整的照应一样, 例如因此, 由于这个原因或因此使困果关系在照应中变得明确. 

与使用精心构造的提取模式集 KZH13 的 shell 名词语料库相比, 我们的方法的缺点在于, 我们人工创建的先行词统一为 $S$ 型. 但是, 在现有数据集中找到的大多数抽象回指的前身是 S 类型. 此外, 我们的模型旨在诱导语义表示, 因此与基于功能的模型相比, 我们期望句法形式的重要性不那么高. 最后, 图 2 中的一般提取模式涵盖了更广泛的回指类型. 

使用这种方法, 我们从 PTB 语料库的 WSJ 部分生成了一个人造的回指句-先行词对数据集, 该数据集使用 Stanford Parser 进行了自动解析. 



### 5 实验设置



### 5.1 数据集

我们对两种类型的回指进行模型评估: (a) shell 名词回指和 (b) 从 ARRAU 中提取的 (代词)名词性抽象回指. 

**a. Shell 名词解析数据集**. 为了实现可比性, 我们使用 Kolhatkar 等人的原始训练 (CSN) 和测试 (ASN) 语料对 shell 名词解析进行训练和评估. 

我们遵循 Kolhatkar 等人的数据准备和评估协议. 

CSN 语料库是使用手动开发的模式从 NYT 语料库中构建的, 从识别后缀 shell 名词 (CSN) 的先行. 





未完...















