## HiGRU: 分级门控循环单元, 用于发声级情绪识别

https://arxiv.org/pdf/1904.04446.pdf



### 摘要

在本文中, 我们解决了对话系统中发声级情感识别的三个挑战: (1) 相同的单词可以在不同的上下文中表达不同的情感; (2) 在一般对话中很少见到某些情绪; (3) 长距离上下文信息难以有效捕获. 因此, 我们提出了一个层次化门控循环单元 (HiGRU) 框架, 该框架具有一个较低的 GRU (用于建模单词级输入) 和一个较高级的 GRU (用于捕获话语级嵌入的上下文). 此外, 我们将框架升级为两个变体, 具有个体特征融合的 HiGRU (HiGRUf) 和具有自我注意和特征融合的 HiGRU (HiGRU-sf), 因此单词/话语水平的个体输入和远程上下文信息可以充分利用. 对三个对话情感数据集 IEMOCAP, Friends 和 EmotionPush 进行的实验表明, 我们提出的 Hi-GRU 模型分别比每个数据集上的最新方法分别至少提高了 8.7%, 7.5% 和 6.0%. 特别是, 通过仅利用 IEMOCAP 中的文本特征, 我们的 HiGRU 模型比具有文本, 视频和音频的三种特征的最新会话存储网络 (CMN) 至少提高了 3.8%. 



### 1 介绍

情感识别是一个重要的人工智能研究主题, 这归因于为人们开发善解人意的机器的潜力. 情绪是跨不同文化的普遍现象, 主要由六种基本类型组成: 愤怒, 厌恶, 恐惧, 幸福, 悲伤和惊奇 (Ekman, 1971, 1992). 

在本文中, 我们专注于文本对话系统, 因为文本特征比音频和视频特征更重要 (Poria et al., 2015, 2017) . 在发声级情绪识别, 发声 (Olson, 1977) 是受呼吸或停顿限制的语音单位, 其目标是在与所指示的情感对话中标记每个发声. 

在此任务中, 我们面临三个挑战: 首先, 同一个单词可以在不同的上下文中传递不同的情感. 例如, 在图 1 中, 单词 "okay" 可以分别传递三种不同的情绪: 愤怒, 中立和欢乐. 诸如喜悦和愤怒之类的强烈情绪可能会以符号 "!" 或 "?" 标识在词中. 为了准确地确定说话者的情绪, 我们需要充分探讨对话的背景. 其次, 在一般对话中很少看到某些情绪. 例如, 人们通常会保持镇定并表现出中性的情绪, 而仅在某些特定情况下, 他们才会表现出强烈的情绪, 例如愤怒或恐惧. 因此, 我们需要对少数情感保持敏感, 同时减轻多数情感的影响. 第三, 在话语/对话中很难有效地捕获远距离上下文信息, 尤其是当测试集中的语音/对话比训练集中的语音/对话更长. 

为了解决这些挑战, 我们为对话系统中的发声级情感识别提出了一个分层的门控制循环单元 (HiGRU) 框架. 更具体地说, HiGRU 由两个级别的双向 GRU 组成, 一个较低级别的 GRU 建模每个语音的单词序列以产生单独的语音嵌入, 另一个较高级别的 GRU 捕获语音的顺序和上下文关系. 我们进一步将拟议的 Hi-GRU 推广为两个变体, 具有个体特征融合的 HiGRU (HiGRU-f) 和具有自注意力和特征融合的 HiGRU (HiGRU-sf). 在 HiGRU-f 中, 将各个输入 (即, 下层 GRU 中的单词嵌入和上层 GRU 中的单个话语嵌入) 与隐藏状态连接起来, 以分别生成上下文词/话语嵌入. 在 HiGRU-sf 中, 自注意层放置在 GRU 的隐藏状态上, 以学习远距离上下文嵌入, 这些上下文嵌入与原始的单个嵌入和隐藏状态相连, 以生成上下文/话语嵌入. 最后, 将上下文话语嵌入发送到完全连接 (FC) 层以确定相应的情感. 为了减轻数据不平衡问题的影响, 我们遵循 (Khosla, 2018) 通过最小化加权分类交叉熵来训练我们的模型. 

我们将我们的贡献总结如下: 

* 我们提出了一个 HiGRU 框架, 以更好地学习个人话语嵌入和话语的上下文信息, 从而更准确地识别情绪. 
* 我们提出了两种渐进式 HiGRU 变体 HiGRU-f 和 HiGRU-sf, 以分别充分融合各个单词/话语级别信息和远距离上下文信息. 
* 我们对三个文本对话情感数据集 IEMOCAP, Friends 和 EmotionPush 进行了广泛的实验. 结果表明, 我们提出了 HiGRU 模型分别比每个数据集上的最新方法分别提高了至少 8.7%, 7.5% 和 6.0%. 特别是, 通过仅利用 IEMOCAP 中的文本特征, 而且还具有视觉和音频特征的会话存储网络 (CMN) 至少提高了 3.8%. 



### 2 相关工作

基于文本的情感识别是一个长期的研究课题 (Wilson et al., 2004; Yang et al., 2007; Medhat et al., 2014). 如今, 由于出色的性能, 深度学习技术已成为主流方法. 一些杰出的模型包括递归自动编码器  (RAEs) (Socher et al., 2011), 卷积神经网络  (CNNs) (Kim, 2014) 和递归神经网络  (RNNs) (Abdul-Mageed and Ungar, 2017). 然而, 这些模型独立地对待文本, 因此无法捕捉对话中话语的相互依赖性  (Kim, 2014; Lai et al., 2015; Grave et al., 2017; Chen et al., 2016; Yang et al., 2016). 为了利用话语的话境信息, 研究人员主要从两个方向进行探索: (1) 在话语中提取语境信息, 或者 (2) 获得更多嵌入在词和话语表征中的信息. 

**上下文信息提取.** RNN 体系结构是捕获数据顺序关系的标准方法.  Poria et al 提出了一种称为 bcLSTM 的双向上下文长期短期记忆 (LSTM) 网络, 以对 CNN 提取的文本特征的上下文进行建模. Hazarika et al. 通过对话式记忆网络 (CMN) 来提高 bcLSTM, 以捕获自我和说话者之间的情感影响, 其中 GRU 用于建模自我影响, 而注意力机制则用于挖掘说话者之间的情感影响. 尽管据报道 CMN 在 IEMOCAP 上的性能优于 bcLSTM (Hazarika et al., 2018), 但对于小型对话数据集而言, 存储网络过于复杂. 

**表征富集.** 多模态特征已被用于丰富话语的表达 (Poria et al., 2015, 2017). 先前的研究表明, 与视觉或听觉特征相比, 文本特征在识别情绪的表现中占主导地位 (Poria et al., 2015, 2017). 最近, 主要由 CNN 提取文本特征以学习单个话语嵌入  (Poria et al., 2015, 2017; Zahiri and Choi, 2018; Hazarika et al., 2018). 但是, CNN 不会在每个发声孔内捕获上下文信息.  另一方面, 已经提出了分层 RNN, 并在常规文本分类任务 (Tang et al., 2015), 对话行为分类  (Liu et al., 2017; Kumar et al., 2018) 和说话人变更方面表现出良好的性能检测 (Meng et al., 2017). 但是, 在对话系统中发声情感识别的任务中并未对其进行很好的探索. 



### 3 方法

话语级情感识别的任务定义如下: 

**定义 1 (话语级情绪识别). ** 假设我们进行了一系列对话, $D = \{ D_{i} \}_{i=1}^{L}$, 其中 $L$ 是对话的数量. 在每一个对话中 $D_{i} = \{ (u_{j}, s_{j}, c_{j}) \}_{j=1}^{N_{i}}$, 是一个 $N_{i}$ 个话语的序列, 其中话语 $u_{j}$ 是由说话者 $s_{j} \in S$ 以一个特定的情绪 $c_{j} \in C$ 说的. 所有说话者都组成了集合 $S$, 集合 $C$ 包含了所有情绪, 例如愤怒, 喜悦, 悲伤和中立. 我们的目标是训练模型 $M$, 以便使用 $C$ 的情感标签尽可能准确地标记每个新话语. 为了解决此任务, 我们提出了一个分层门控循环单元 (HiGRU) 框架, 并扩展了两个渐进式变体, 即具有单个特征融合的 HiGRU (HiGRU-f) 和具有自注意和特征融合的 HiGRU (HiGRU-sf) (如图 2 所示). 



### 3.1 HiGRU: 层次 GRU

vanilla HiGRU 由两级 GRU 组成: 较低级的双向 GRU 通过在语音中建模单词序列来学习半日个语音嵌入, 而较高级的双向 GRU 通过对话语语音中的单词序列建模来学习上下文语音嵌入. 

**单个话语的嵌入.** 对于 $D_{i}$ 中的第 $j$ 句话 $u_{j} = \{ w_{k} \}_{k=1}^{M_{j}}$, 其中 $M_{j}$ 是话语 $u_{j}$ 中的词语数. 单个单词的嵌入以相应顺序 $\{ e(w_{k}) \}_{k=1}^{M_{j}}$ 被馈入低级双向 GRU 以从两个相反的方向学习单个话语嵌入. 

$$\begin{aligned} \overrightarrow{h_{k}} &= GRU (e(w_{k}), \overrightarrow{h_{k-1}}), \quad &(1) \\ \overleftarrow{h_{k}} &= GRU (e(w_{k}), \overleftarrow{h_{k+1}}). \quad &(2) \end{aligned}$$ 

两个隐状态 $\overrightarrow{h_{k}}$ 和 $\overleftarrow{h_{k}}$ 被串联为 $h_{S} = [\overrightarrow{h_{k}}; \overleftarrow{h_{k}}]$ 通过 tanh 激活函数在线性变换上生成 $w_{k}$ 的上下文词嵌入: 

$$\begin{aligned} e_{c}(w_{k}) = \text{tanh}(W_{w} \cdot h_{S} + b_{w}), \quad (3) \end{aligned}$$ 

其中: $W_{w} \in \mathbb{R}^{d_{1} \times 2 d_{1}}$ , $b_{w} \in \mathbb{R}^{d_{1}}$ 是模型参数. $d_{0}$ 和 $d_{1}$ 分别是词嵌入和低级 GRU 的隐状态层的维度. 

然后, 通过对话语中的上下文词嵌入进行最大池化来获得单个话语嵌入: 

$$\begin{aligned} e(u_{j}) = \text{maxpool}(\{ e_{c}(w_{k}) \}_{k=1}^{M_{j}}). \quad (4) \end{aligned}$$ 



**上下文话语嵌入. ** 对于第 $i$ 个对话 $D_{i} = \{ (u_{j}, s_{j}, c_{j}) \}_{j=1}^{N_{i}}$, 训练得到的单个话语嵌入, $\{ e(u_{j}) \}_{j=1}^{N_{i}}$, 被馈入高级双向 GRU 以捕获对话中单个话语的顺序和上下文关系: 

$$\begin{aligned} \overrightarrow{H_{j}} &= GRU (e(u_{j}), \overrightarrow{H_{j-1}}), \quad &(5) \\ \overleftarrow{H_{j}} &= GRU (e(u_{j}), \overleftarrow{H_{j+1}}). \quad &(6) \end{aligned}$$ 

这里, 高级 GRU 的隐状态由 $H_{j} \in \mathbb{R}^{d_{2}}$ 表示, 以区别于用 $h_{k}$ 表示的较低级别 GRU 中学习到的知识. 因此, 我们可以通过以下方式获得上下文话语嵌入: 

$$\begin{aligned} e_{c}(u_{j}) = \text{tanh}(W_{u} \cdot H_{S} + b_{u}), \quad (7) \end{aligned}$$ 

其中: $H_{S} = [\overrightarrow{H_{j}}; \overleftarrow{H_{j}}]$, $W_{u} \in \mathbb{R}^{d_{2} \times 2 d_{2}}$ 和 $b_{u} \in \mathbb{R}^{d_{2}}$ 是模型参数, $d_{2}$ 是高级 GRU 的隐状态的维度. 因为情绪通过句子级别识别, 学习到的单个话语嵌入 $e_{c}(u_{j})$ 被直接馈入 $FC$ 全连接层, 紧跟一个 $softmax$ 函数来确定相关的情绪标签: 

$$\begin{aligned} \hat{y}_{j} = \text{softmax} (W_{fc} \cdot e_{c}(u_{j}) + b_{fc}), \quad (8) \end{aligned}$$ 

其中: $\hat{y}_{j}$ 是所有情绪的预测向量, $W_{fc} \in \mathbb{R}^{|C| \times d_{2}}$, $b_{fc} \in \mathbb{R}^{|C|}$. 



### 3.2 HiGRU-f: HiGRU + Individual Features Fusion (个别特征融合)

vanilla HiGRU 包含两个主要问题: (1) 各个单词/话语嵌入随着层的堆叠而被稀释; (2) 高级 GRU 倾向于从多数情绪收集更多上下文信息, 这会降低整体模型的性能. 

为了解决这两个问题, 我们建议将单个单词/话语嵌入与 GRU 的隐藏状态融合, 以增强每个单词/话语在上下文嵌入中的信息. 此变体称为 HiGRU-f, 代表具有个别特征融合特征的 HiGRU. 因此, 较低级别的 GRU 可以保留单个单词的嵌入, 而较高级别的 GRU 可以减轻多数情绪的影响, 并获得针对不同情绪的更精确的话语表示. 具体而言, 上下文嵌入将更新为: 

$$\begin{aligned} e_{c}(w_{k}) &= \text{tanh}(W_{w} \cdot h_{S}^{f} + b_{w}), \quad &(9) \\ e_{c}(u_{j}) &= \text{tanh}(W_{u} \cdot H_{S}^{f} + b_{u}). \quad &(10) \end{aligned}$$ 

其中: $W_{w} \in \mathbb{R}^{d_{1} \times (d_{0} + 2 d_{1})}$, $W_{u} \in \mathbb{R}^{d_{2} \times (d_{1} + 2 d_{2})}$, $h_{S}^{f} = [\overrightarrow{h_{k}}; e(w_{k}); \overleftarrow{h_{k}}]$, $H_{S}^{f} = [\overrightarrow{H_{j}}; e(u_{j}); \overleftarrow{H_{j}}]$. 





### 3.3 HiGRU-sf: HiGRU + 自我注意和特征融合

另一个有有挑战性的问题是提取长序列的上下文信息, 尤其是测试集中的序列比训练集中的序列更长的序列信息 (Bahdanau et al., 2014) . 为了充分利用全局上下文信息, 我们在 HiGRU 的隐藏状态上放置了一个自我注意层, 并将注意输出与单个单词/话语嵌入和隐藏状态相融合, 以学习上下文单词/话语嵌入. 因此, 该变体被称为 HiGRU-sf, 代表具有自我注意力和特征融合的 HiGRU. 

特别地, 我们分别对向前和向后隐藏状态应用自我注意, 以分别产生左上下文嵌入, $h_{k}^{l}(H_{j}^{l})$, 和右上下文嵌入 $h_{k}^{r}(H_{j}^{r})$. 这使我们可以在当前步骤中沿两个相反的方向收集唯一的全局上下文信息, 并产生如下计算的相应上下文嵌入: 

$$\begin{aligned} e_{c}(w_{k}) &= \text{tanh}(W_{w} \cdot h_{S}^{sf} + b_{w}), \quad &(11) \\ e_{c}(u_{j}) &= \text{tanh}(W_{u} \cdot H_{S}^{sf} + b_{u}). \quad &(12) \end{aligned}$$ 

其中: $W_{w} \in \mathbb{R}^{d_{1} \times (d_{0} + 4 d_{1})}$, $W_{u} \in \mathbb{R}^{d_{2} \times (d_{1} + 4 d_{2})}$, $h_{S}^{sf} = [h_{k}^{l}; \overrightarrow{h_{k}}; e(w_{k}); \overleftarrow{h_{k}}; h_{k}^{r}]$, $H_{S}^{sf} = [H_{j}^{l}; \overrightarrow{H_{j}}; e(u_{j}); \overleftarrow{H_{j}}; H_{j}^{r}]$. 



**自我注意 (SA). ** 自我注意机制是一种有效的非递归体系结构, 用于计算一个输入与所有其他输入之间的关系, 并已成功应用于各种自然语言处理应用中, 例如阅读理解 (Hu et al., 2018) 和神经机器解译 (Vaswani et al., 2017).  图 3 显示了 GRU 的前向隐藏状态上的点积 SA, 以学习左上下文 $H_{k}^{l}$. 注意矩阵中的每个元素的计算公式为: 

$$f(\overrightarrow{h_{k}}, \overrightarrow{h_{p}}) = \left\{ \begin{aligned} & \overrightarrow{h_{k}}^{T} \overrightarrow{h_{p}}, \quad & \text{if} \space k,p \le M_{j}, \\ &- \infty, & \text{otherwise}. \end{aligned} \right. \quad (13)$$ 

然后应用注意遮罩以免除序列输入和填充之间的内部注意. 在每个步骤中, 通过所有前向隐藏状态的加权总和来计算对应的左上下文 $h^{l}_{k}$: 

$$\begin{aligned} h^{l}_{k} = \sum_{p=1}^{M_{j}}{a_{kp} \overrightarrow{h_{p}}}, \quad a_{kp} = \frac{\exp{(f(\overrightarrow{h_{k}}, \overrightarrow{h_{p}}))}}{\sum_{p^{'}=1}^{M_{j}}{\exp{(f(\overrightarrow{h_{k}}, \overrightarrow{h_{p'}}))}}}, \quad (14) \end{aligned}$$ 

其中 $a_{kp}$ 是 $\overrightarrow{h_{p}}$  的权重以包含到 $h^{l}_{k}$ 中. 右上下文 $h^{r}_{k}$ 可以通过类似的方法计算. 



### 3.4 模型训练

继 (Khosla, 2018) 在 EmotionX 共享任务中获得最佳性能 (Hsu and Ku, 2018) 之后, 我们最小化了所有对话的每个语句的加权分类交叉熵, 以优化模型参数: 

$$loss = -\frac{1}{\sum_{i=1}^{L}{N_{i}}} \sum_{i=1}^{L} \sum_{j=1}^{N_{i}}{\omega(c_{j})} \sum_{c=1}^{|C|}{y_{j}^{c} \log_{2}(\hat{y}_{j}^{c})}, \quad (15) $$ 

其中: $y_{j}$ 是原始 one-hot 向量的情绪标签, $y_{j}^{c}$ 和 $\hat{y}_{j}^{c}$ 是与类别 $c$ 相关的 $y_{j}$ 和 $\hat{y}_{j}$ 的元素. 

类似于 (Khosla, 2018), 我们应用损失权重 $\omega(c_{j})$, 它与类别 $c_{j}$ 中的训练话语数量成反比, 用 $I_{c}$ 表示, 即为少数群体分配更大的损失权重, 以缓解数据不平稀问题. 不同之处在于我们添加了一个常数 $\alpha$ 来调整分布的平滑度. 有: 

$$\begin{aligned} \frac{1}{\omega(c)} = \frac{I_{c}^{\alpha}}{\sum_{c^{'}}^{|C|}{I_{c^{'}}^{\alpha}}} \quad (16) \end{aligned}$$ 









































































