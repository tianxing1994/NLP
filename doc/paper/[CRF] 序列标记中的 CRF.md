## 序列标记中的 CRF

序列标记模型中所使用的 CRF, 实际上是隐马尔可夫链. 



### 1. 观测转换为隐状态

在隐马尔可夫链中, 需要计算各隐状态中各观测发生的概率. 此处基于深度学习的方法使我们不能从训练样本中统计得出这些 **发射概率**, 因此, 我们定义由观测推算出当前节点为各隐状态的概率, 等式如下. 

$$\begin{aligned} y = \text{softmax}(x \cdot W_{d \times c} + b) \quad (1) \end{aligned}$$ 

其中 $x \in \mathbb{R}^{d}$ 是输入观测值, $W \in \mathbb{R}^{d \times c}$, $b \in \mathbb{R}^{c}$ 是计算矩阵, $y \in \mathbb{R}^{c}$ 表示当前观测由各隐状态发射而来的概率, 其中 $d$ 代表模型输出观测的维度. $c$ 代表了隐状态数量, (即, 类别数量). 



### 2. 定义状态转移矩阵

定义状态转移矩阵 $T_{c \times c}$, 其中第 $i$ 行, 第 $j$ 列的值表示, 隐状态从 $i$ 转换到 $j$ 的概率 (显然, $T_{c \times c}$ 的每一行的和应该为 1, 为此我们本应该定义矩阵 $\hat{T}_{c \times c}$ 然后通过 $\text{softmax}$ 将其每一行归一化, 此处暂不作讨论). 那么, 假设时间 $t-1$ 时的隐状态表示为 $y_{t-1}$ ($e \in \mathbb{R}^{c}$, 表示 $t-1$ 时刻, 各隐状态发生的概率), 则结合状态转移矩阵可以计算 **转移概率**: 

$$\begin{aligned} y_{t} = y_{t-1} \cdot T_{c \times c} \quad (2) \end{aligned}$$ 

其中: $y_{t}$, $y_{t-1}$ 都是形状为 $(1, c)$ 的矩阵. $y_{t}$ 的第 1 个值由 $y_{t-1}$ 与 $T_{c \times c}$ 的第 1 列内积得来, 可以理解为 $t$ 时刻为隐状态 $0$ 的概率为 $t-1$ 时刻各隐状态发生的概率转移到 $t$ 时刻时隐状态为 $0$ 的概率之和. 



### 3. 隐状态的概率

由发射概率和转移概率 (等式 1 和等式 2), 我们可以直算 $t$ 时刻的隐状态概率: 

$$\begin{aligned} y_{t} = (y_{t-1} \cdot T_{c \times c}) \times\text{softmax} (x_{t} \cdot W_{d \times c} + b) \quad (3) \end{aligned}$$ 





### 4. 概率取 log 对数, 累乘变累加

众所周知, 概率在 0 到 1 之间, 累乘计算很可能会使其值越来越小, 而使计算变得不切实际. 因此, 我们讨论如何将上述中的概率转换成 $\log$ 对数形式. 



### 4.1 观测转隐状态

等式 1: 

$$\begin{aligned} y &= \text{softmax}(x \cdot W_{d \times c} + b) \end{aligned}$$ 

我们很难直接将上式转换成对数形式, 但我们令模型直接学习 $\log(y)$ 的表示, 即令: 

$$\begin{aligned} e &= x \cdot W_{d \times c} + b \\ &= \log(y) \end{aligned} \quad (4) $$ 



### 4.2 转移矩阵

如前所述, 转移 $M_{c \times c}$ 本应该满足: 其每一行的和应该为 1. 前面我们没有去解决这个问题. 而此处, 我们令模型学习概率的对数表示, 则其不再需要满足该条件. 

我们用:

*  $e_{t-1}$ 表示 $t-1$ 时刻各隐状态概率的对数表示 $\log(p)$. 
* $M_{c \times c}$ 表示各隐状态之间的概率转换关系. 其中, 第 $i$ 行, 第 $j$ 列的值表示, 隐状态从 $i$ 转换到 $j$ 的概率的对数 $\log(p)$. 

则 $t$ 时刻的 **转移概率** 表示为: 

$$\begin{aligned} e_{t} &= \underbrace{\text{logsumexp}}_{\text{axis=0}}(e_{t-1}^{T} + M_{c \times c}) \quad (5) \end{aligned}$$ 

其中: 

*  $e_{t}$, $e_{t-1}$ 都是形状为 $(1, c)$ 的矩阵. 应注意, 我们对 $e_{t-1}$ 进行了转置, 在与 $M_{c \times c}$ 进行求和的时候自动广播到 $(c, c)$ 形状. 
* $e_{t-1}^{T} + M_{c \times c}$ 后, 得到的是大小  $(c, c)$ 的矩阵, 其第 $i$ 行, 第 $j$ 列的值表示 $t-1$ 时刻到 $t$ 时刻的隐状态由 $i$ 转移到 $j$ 的概率的对数. 我们对其在 $\text{axis}=0$ 维求和, 得到 $t$ 时刻为各隐状态的概率的对数. 



### 4.3 $t$ 时刻的隐状态

$t$ 时刻的隐状态可由观测和状态转移共同计算得出: 

$$\begin{aligned} e_{t} &= (x_{t} \cdot W_{d \times c} + b) + \underbrace{\text{logsumexp}}_{\text{axis=0}}(e_{t-1}^{T} + M_{c \times c}) \end{aligned} \quad (6)$$ 

考虑到 $(e_{t-1}^{T} + M_{c \times c})$ 中, 第 $i$ 行, 第 $j$ 列的值表示 $t-1$ 时刻到 $t$ 时刻的隐状态由 $i$ 转移到 $j$ 的概率的对数. 则 $(x_{t} \cdot W_{d \times c} + b) + (e_{t-1}^{T} + M_{c \times c})$ 表示观测和状态转移共同计算得出的 $t-1$ 时刻到 $t$ 时刻的隐状态由 $i$ 转移到 $j$ 的概率的对数. 因此, $t$ 时刻, 各隐状态的概率的对数也可表示为: 

$$\begin{aligned} e_{t} &= \underbrace{\text{logsumexp}}_{\text{axis=0}}((x_{t} \cdot W_{d \times c} + b) + e_{t-1}^{T} + M_{c \times c}) \quad (7) \end{aligned}$$ 





### 5 最大似然估计

由于我们无法计算出所有可能的路径的概率, 因此隐马尔可夫模型一般采用动态归化算法求解最佳路径. 

最大似然估计分为两部分: 

1. 发射概率的得分. 
2. 转移概率的得分. 



### 5.1 发射概率得分

根据**等式 4**, 我们可以计算出 $e = [ e_{1}, e_{2}, \cdots , e_{T} ]$ 其中 $e_{i}$ 表示 $i$ 时刻的隐状态发生的概率的对数. 

根据真实标签 $Y_{true}$, 可以计算真实路径在发射概率上的得分. 如下: 

$$\begin{aligned} \text{transmit_score} = \sum{Y_{true} \times e} \end{aligned}$$. 

注意: $Y_{true}$ 是独热编码格式, 形状为 $(T, c)$, $e$ 的形状也是 $(T, c)$ . $e$ 中的值表示概率的对数, 对数求和等式概率累乘. 因此 $\text{transmit_score}$ 实际上表示的是在没有状态转移影响时, 由观测发射出正确路径 $Y_{true}$ 的概率的对数. 



### 5.2 转移概率得分

正如等式 6 表示的, 路径的得分是由发射概率和转移概率共同表示的. 现在我们计算在只考虑隐状态转移的情况下, 正确路径被得出的概率. 

此处我们只需要考虑正确路径中, $t-1$ 时刻的隐状态和 $t$ 时刻的隐状态, 并从概率转移矩阵中取出该状态转移的概率值即可. 

令: 

* $Y_{1} = \text{expand_dims}(T_{true}[:-1, :], \text{axis}=2)$, 形状为 $(T-1, c, 1)$. 
* $Y_{2} = \text{expand_dims}(T_{true}[1:, :], \text{axis}=1)$, 形状为 $(T-1, 1, c)$. 
* $Y = Y_{1} \times Y_{2}$, 形状为 $(T-1, c, c)$. 
* $M = \text{expand_dims}(M_{c \times c}, \text{axis}=0)$, 形状为 $(1, c, c)$. 



$$\begin{aligned} \text{transfer_score} = \sum{Y \times M} \end{aligned}$$. 

注意: 此处, 同样地, 状态转移矩阵中的值表示概率的对数, 对数求和表示概率累乘. 因此, $\text{transfer_score}$ 表示的是仅考虑概率转移时, 正确路径发生的概率. 



### 5.3 正确路径发生的概率与损失函数

目前我们计算出了正确路径 $Y_{true}$ 发生的概率. 但存在一个问题: 在最大似然计算时, 只要 **发射概率** 和 **转移概率** 中的每一个值都变大, 则正确路径 $Y_{true}$ 发生的概率就会变大. 这显然不符合模型训练的目的. 因此, 我们需要计算, 在所有可能路径中, 其它路径发生的概率不能变大, 而正确路径 $Y_{true}$ 的概率变大. 

要计算所有可能路径的概率是无法实现的, 因此, 另一个替代的方法是, 我们计算出最后一个结点分别为 $0, 1, \cdots c-1$ 的 $c$ 个最佳路径. 将这 $c$ 个最佳路径视作所有可能的路径. 因此我们根据等式 7, 递归计算出 $e_{T}$. 

得出正确路径的绝对概率为: 

$$\begin{aligned} \log(P) =  (\text{transmit_score} + \text{transfer_score}) - \text{logsumexp}(e_{T}) \end{aligned}$$. 

其中: $\text{transmit_score}$, $\text{transfer_score}$, $e_{T}$ 都表示概率的对数. 因此, 采用了函数 $\text{logsumexp}$ 来求概率和, 同时, 对数相减等于概率相除. 

一般函数采用最小化优化, 因此将上式转换为负对数, 如下: 

$$\begin{aligned} - \log(P) &=  - ((\text{transmit_score} + \text{transfer_score}) - \text{logsumexp}(e_{T})) \\ &= \text{logsumexp}(e_{T}) - (\text{transmit_score} + \text{transfer_score}) \end{aligned}$$. 



### 5.4 训练精度

训练的精度度量应由根据**等式 7**, 计算出的 $e = [ e_{1}, e_{2}, \cdots , e_{T} ]$ 经过维特比算法解码, 求得最优路径后与 $Y_{true}$ 计算各节点的准确率. 











































