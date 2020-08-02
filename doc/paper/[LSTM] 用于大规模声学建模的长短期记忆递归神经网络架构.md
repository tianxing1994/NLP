## 用于大规模声学建模的长短期记忆递归神经网络架构

https://research.google.com/pubs/archive/43905.pdf



### 摘要

长短期记忆 (LSTM) 是一种特定的递归神经网络 (RNN) 体系结构, 旨在比常规 RNN 更准确地对时间序列及其长期依赖性进行建模. 在本文中, 我们探索了用于语音识别中大规模声学建模的 LSTM RNN 体系结构. 我们最近表明, 考虑到在单个机器上训练的中等大小的模型, 对于声学建模, LSTM RNN 比 DNN 和常规 RNN 更有效. 在这里, 我们介绍了在大型机器集群上使用异步随机梯度下降优化对 LSTM RNN 进行首次分布式训练. 我们显示了一个两层深的 LSTM RNN, 其中每个 LSTM 层都有一个线性递归投影层, 可以超过最新的语音识别性能. 该体系结构比其他模型更有效地利用模型参数, 收敛迅速, 并且优于具有更多参数数量级的深度前馈神经网络. 



### 1. 介绍

语音是一个复杂的时变信号, 在不同的时标范围内具有复杂的相关性. 递归神经网络(RNN)包含循环连接, 这使它们成为比前馈神经网络更强大的工具来建模比类序列数据. RNN 已在序列标记和预测任务 (例如手写识别和语言建模) 中取得了巨大的成功. 然而, 在用于语音识别的声学模型中, 深度神经网络 (DNN) 是最先进的技术, 最近, RNN 除了小规模的电话识别任务之外, 几乎没有受到关注, Robinson 的工作是一个明显的例外, Graves 和 Sak. 

通过在声学框架的固定大小的滑动窗口上进行操作, DNN 只能提供有限的时间建模. 他们只能在窗口内对数据建模, 不适合处理不同的语音速率和长期依赖性. 相比之外, 递归神经网络包含循环, 该循环将前一时间步长的网络激活作为网络的输入, 以影响当前时间步长的预测. 这些激活存储在网络的内部状态中, 该内部状态原则上可以保存长期的时间上下文信息. 这种机制允许 RNN 在输入序列历史上利用动态变化的上下文窗口, 而不是像前馈网络所使用的固定大小窗口那样利用静态窗口. 特别是, 克服了 RNN 建模的某些缺点的长短期记忆 (LSTM) 体系结构, 在概念上对声学建模具有吸引力. 

LSTM 和常规 RNN 已成功应用于各种序列预测和序列标记任务. 在语言建模中, 传统的 RNN 与标准 ngram 模型相比大大降低了复杂度, 而 LSTM RNN 模型则比传统的 RNN LM 有所改善. 已经证明, LSTM 模型在学习上下文无关和上下文敏感的语言方面比 RNN 表现更好. 为了在 TIMIT 语音数据库上对声帧进行语音标记, 已经提出了双向 LSTM (BLSTM) 网络, 该网络在两个方向上对输入序列进行操作以决定当前的输入. 对于在线和离线手写识别, 已证明 BLSTM 网络与连接主义者的时间分类 (CTC) 层一起使用并从未分段的序列数据中进行训练, 其性能优于基于最新的隐马尔可夫模型 (HMM) 的系统. 已经提出了具有深层 BLSTM 网络的类似技术来执行基于字形的语音识别. 还提出了 BLSTM 网络用于在多流框架中进行连续会话语音识别的音素预测. 在架构方面, 随着 DNN 在声学建模中的成功, 结合 CTC 输出层和预测电话序列的 RNN 换能器的深度 BLSTM RNN 已被证明可以达到最佳状态 TIMIT 数据库上具有先进技术的电话识别准确性. 

最近, 在混合语音识别方法中, 深度 BLSTM RNN 比 DNN 表现更好. 使用混合方法, 我们最近发现, 考虑到在单个机器上训练的中等大小的模型, 具有递归投影层的 LSTM 架构在大词汇量语音识别方面要优于 DNN 和常规 RNN. 在本文中, 我们探索了使用分布式训练进行大规模声学建模的 LSTM RNN 体系结构. 我们显示了一个两层深的 LSTM RNN, 其中每个 LSTM 层都有一个线性递归投影层, 其性能优于使用深度前馈神经网络的强基线系统, 该系统具有更多数量级的参数. 



### LSTM 网络结构

#### 2.1. 卷积 LSTM

LSTM 在循环隐藏层中包含称为存储块的特殊单元. 除了称为控制信息流的称为门的特殊乘法单元之外, 存储块还包 含具有自连接的存储单元, 这些自连接存储网络的时间状态. 原始体系结构中的每个存储块都包含一个输入门和一个输出门. 输入门控制输入激活进入存储单元的流程. 输出门控制单元激活进入网络其余部分的输出流. 然后, 忘记门被添加到存储块. 这解决了 LSTM 模型的一个缺点, 即 LSTM 模型无法处理未分段为子序列的连续输入流. 遗忘门会先缩放单元的内部状态, 然后再通过单元的自循环连接将其作为输入添加到单元中, 从而适应性地忘记或重置单元的内存. 另外, 现代的 LSTM 架构包含从内部单元到同一单元中的门的窥孔连接, 以了解输出的精确时序. 

LSTM 网络通过使用以下等式从 $t=1$ 到 $t=T$, 迭代地计算网络单元激活来计算从输入序列 $x = (x_{1}, x_{2}, \cdots x_{T})$ 到输出序列 $y=(y_{1}, y_{2}, \cdots , y_{T})$ 的映射. 

$$\begin{aligned} i_{t} = \sigma (W_{ix}x_{t} + W_{im} m_{t-1} + W_{ic}c_{t-1} + b_{i}) \quad (1) \\ f_{t} = \sigma(W_{fx}x_{t} + W_{fm}m_{t-1} + W_{fc}c_{t-1} + b_{f}) \quad (2) \\ c_{t} = f_{t} \odot c_{t-1} + i_{t} \odot g(W_{cx}x_{t} + W_{cm}m_{t-1} + b_{c}) \quad (3) \\ o_{t} = \sigma(W_{ox}x_{t} + W_{om}m_{t-1} + W_{oc}c_{t} + b_{o}) \quad (4) \\ m_{t} = o_{t} \odot h(c_{t}) \quad (5) \\ y_{t} = \phi(W_{ym}m_{t} + b_{y}) \quad (6) \end{aligned}$$ 

其中: $W$ 项为权重矩阵. $W_{ix}$ 是输入门到输入的权重矩阵, $W_{ic}, W_{fc}, W_{oc}$ 是用于窥孔连接的对角线权重矩阵. $b$ 项表示偏置矢量 ($b_{i}$ 是输入门偏置矢量), $\sigma$ 是逻辑 $sigmoid$ 函数, 而 $i, f, o, c$ 分别是输入门, 忘记门, 输出门和单元激活向量. 它们的大小都与单元输出激活向量 $m$ 相同, $\odot$ 是向量的按元素乘积 (不是内积), $g$ 和 $h$ 是单元输入和单元输出激活函数, 在本文中使用的是 $tanh$, $\phi$ 是网络输出激活函数 $softmax$ 在本文中. 



```python
# https://blog.csdn.net/omnispace/article/details/78415204
class LSTMCell(object):
    def __init__(self, train_data, train_label, hidden_units=64, num_units=128):
        """
        :param train_data:
        :param train_label:
        :param hidden_units: 隐藏层的单元数量.
        :param num_units: 输出向量的维度.
        """
        with tf.variable_scope(name_or_scope="input", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as input_layer:
            self.ix, self.im, self.ic, self.ib = self._generate_w_b(
                x_weights_size=(Config.vocabulary_size, hidden_units),
                m_weights_size=(hidden_units, hidden_units),
                c_weights_size=(hidden_units, hidden_units),
                biases_size=(1, hidden_units)
            )

        with tf.variable_scope(name_or_scope="forget",
                               initializer=tf.truncated_normal_initializer(-0.1,
                                                                           0.1)) as forget_layer:
            self.fx, self.fm, self.fc, self.fb = self._generate_w_b(
                x_weights_size=(Config.vocabulary_size, hidden_units),
                m_weights_size=(hidden_units, hidden_units),
                c_weights_size=(hidden_units, hidden_units),
                biases_size=(1, hidden_units)
            )

        with tf.variable_scope(name_or_scope="memory", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as update_layer:
            self.cx, self.cm, _, self.cb = self._generate_w_b(
                x_weights_size=(Config.vocabulary_size, hidden_units),
                m_weights_size=(hidden_units, hidden_units),
                c_weights_size=(hidden_units, hidden_units),
                biases_size=(1, hidden_units)
            )

        with tf.variable_scope(name_or_scope="output", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as output_layer:
            self.ox, self.om, self.oc, self.ob = self._generate_w_b(
                x_weights_size=(Config.vocabulary_size, hidden_units),
                m_weights_size=(hidden_units, hidden_units),
                c_weights_size=(hidden_units, hidden_units),
                biases_size=(1, hidden_units)
            )

        with tf.variable_scope(name_or_scope="output", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as y_layer:
            self.ym = tf.get_variable(name="weights", shape=(hidden_units, num_units))
            self.yb = tf.get_variable(name="biases", shape=(hidden_units, num_units))

        self.w = tf.Variable(tf.truncated_normal(shape=(hidden_units, Config.vocabulary_size), mean=-0.1, stddev=0.1))
        self.b = tf.Variable(tf.zeros(shape=(Config.vocabulary_size,)))

        self.saved_output = tf.Variable(tf.zeros(shape=(Config.batch_size, hidden_units)), trainable=False)
        self.saved_state = tf.Variable(tf.zeros(shape=(Config.batch_size, hidden_units)), trainable=False)

        self.train_data = train_data
        self.train_label = train_label

    @staticmethod
    def _generate_w_b(x_weights_size, m_weights_size, c_weights_size, biases_size):
        x_w = tf.get_variable(name="x_weights", shape=x_weights_size)
        m_w = tf.get_variable(name="m_weights", shape=m_weights_size)
        c_w = tf.get_variable(name="c_weights", shape=c_weights_size)
        b = tf.get_variable(name="biases", shape=biases_size, initializer=tf.constant_initializer(0.0))
        return x_w, m_w, c_w, b

    def _run(self, x, memory, cell_state):
        """
        :param x: shape=(batch, vocab_size)
        :param memory: shape=(batch, hidden_units)
        :param cell_state: shape=(batch, hidden_units)
        :return:
        """

        input_gate = tf.sigmoid(tf.matmul(x, self.ix) + tf.matmul(memory, self.im) + tf.matmul(cell_state, self.ic) + self.ib)
        forget_gate = tf.sigmoid(tf.matmul(x, self.fx) + tf.matmul(memory, self.fm) + tf.matmul(cell_state, self.fc) + self.fb)
        update = tf.matmul(x, self.cx) + tf.matmul(memory, self.cm) + self.cb
        cell_state = forget_gate * cell_state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(x, self.ox) + tf.matmul(memory, self.om) + self.ob)
        memory = output_gate * tf.tanh(cell_state)
        y = tf.sigmoid(tf.matmul(memory, self.ym) + self.yb)
        return y, memory, cell_state

    def loss_func(self):
        y_list = list()
        memory = self.saved_output
        cell_state = self.saved_state
        for x in self.train_data:
            y, _, _ = self._run(x, memory, cell_state)
            y_list.append(y)

        with tf.control_dependencies([
            self.saved_output.assign(memory),
            self.saved_state.assign(cell_state)
        ]):
            logits = tf.concat(y_list, axis=0)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.concat(self.train_label, axis=0),
                    logits=logits
                )
            )

            prediction = tf.nn.softmax(logits)
        return logits, loss, prediction


```





#### 2.2. 深度 LSTM

与具有更深架构的 DNN 一样, 深度 LSTM RNN 已成功用于语音识别. 深度 LSTM RNN 通过堆叠多个 LSTM 层来构建. 请注意, 从某种意义上说, LSTM RNN 可以看作是前馈神经网络, 并且每层共享相同的模型参数, 它们可以在时间上展开, 因此它们已经是深度结构. 可以看到, 模型的输入像 DNN 一样经过多个非线性层, 但是给定时间点的特征仅在贡献该时间点的输出之前由单个非线性层处理. 因此, 深度 LSTM RNN 中的深度还有其他含义. 给定时间步长的网络输入除了通过时间和 LSTM 层传播外, 还经过多个 LSTM 层. 有人认为, RNN 中的较深层允许网络在输入上以不同的时间尺度学习. 与标准 LSTM RNN 相比, 深度 LSTM RNN 提供了另一个好处: 通过将参数分布在多层空间上, 它们可以更好地利用参数. 例如, 不是将标准模型的内存大小增加 2 倍, 而是可以将 4 层具有大致相同数量的参数. 这导致输入在每个时间步长上经历更多的非线性运算. 





#### 2.3. LSTMP - 带有递归投影层的 LSTM

标准的 LSTM RNN 架构具有输入层, 循环 LSTM 层和输出层. 输入层连接到 LSTM 层. LSTM 层中的循环连接直接从单元输出单元到单元输入单元, 输入门, 输出门和忘记门. 单元输出单元也连接到网络的输出层. 可以将标准 LSTM 网络中每个存储块中有一个单元的参数 N 的总数 (biases 偏差不计) 可以计算为: $N = n_{c} \times n_{c} \times 4 + n_{i} \times n_{c} \times 4 + n_{c} \times n_{o} + n_{c} \times 3$, 其中 $n_{c}$ 是记忆单元的数量 (在这种情况下为存储块数量), $n_{i}$ 是输入单元数, $n_{o}$ 是输出单元数. 使用随机梯度下降 (SGD) 优化技术按权重和时间步学习 LSTM 模型的计算复杂度为 $O(1)$. 因此, 每个时间步的学习计算复杂度为 $O(N)$. 输入数量适中的网络的学习时间由 $n_{c} \times (4 \times n_{c} + n_{o})$ 因子决定. 对于需要大量输出单元和大量存储单元来存储时间上下文信息的任务, 学习 LSTM 模型在计算上变得昂贵. 

作为标准体系结构的替代方案, 我们提出了长期长短期记忆计划 (LSTMP) 体系结构, 以解决学习 LSTM 模型的计算复杂性. 如图 1 所示, 该体系结构在 LSTM 层之后具有单独的线性投影层. 现在, 循环连接从该循环投影层连接到 LSTM 层的输入. 网络输出单元连接到该循环层. 此模型中的参数数量为: $n_{c} \times n_{r} \times 4 + n_{i} \times n_{c} \times 4 + n_{r} \times n_{o} + n_{c} \times n_{r} + n_{c} \times 3$, 其中 $n_{r}$ 是循环投影层中的单位数. 这种情况下. 模型大小和学习计算复杂度由 $n_{r} \times (4 \times n_{c} + n_{o})$ 因子决定. 因此, 这允许我们将参数的数量减少 $\frac{n_{r}}{n_{c}}$. 通过设置 $n_{r} < n_{c}$, 我们可以增加模型内存 $(n_{c})$, 并且仍然能够控制循环连接和输出层中的参数数量.  

使用提出的 LSTMP 体系结构, 网络单元激活的方程式稍有变化, 将 $m_{t-1}$ 激活向量替换为 $r_{t-1}$, 并添加了以下内容. 

$$\begin{aligned} r_{t} = W_{rm}m_{t} \quad (7) \\ y_{t} = \phi (W_{yr}r_{t} + b_{y}) \quad (8) \end{aligned}$$

其中: $r$ 表示循环单元激活. 





#### 2.4. 深度 LSTMP

与深度 LSTM 相似, 我们提出了深度 LSTMP, 其中将多个 LSTM 层堆叠在一起, 每个 LSTM 层分别具有单独的循环投影层. LSTMP 允许独立于输出层和循环连接来增加模型的内存. 但是, 我们注意到增加内存大小会通过记忆输入序列数据而使模型更容易过度拟合. 我们知道, 随着深度的增加, DNN 可以更好地推广到看不见的示例. 深度使模型更难于过拟合训练数据, 因为网络的输入需要通过许多非线性函数. 出于这种动机, 我们尝试了深度 LSTMP 架构, 其目的是增加模型的内存大小和泛化能力. 







### 3. 分布式训练: 通过并行化扩展到大型模型 

我们选择在多核 CPU 而非 GPU 上实现 LSTM RNN 架构. 该决定是基于 CPU 的相对较简单的实现复杂性, 调试的简例性以及使用由商用硬件制成的群集的能力. 对于矩阵运算, 我们使用了特征矩阵库. 这个模板化的 C++ 库使用矢量化指令为 CPU 上的矩阵运算提供了有效实现. 我们使用 SIMD 指令在矩阵上实现了激活函数和梯度计算, 从而受益于并行化. 

我们使用经过时间截断的反向传播 (BPTT) 学习算法来计算训练语音的短子序列上的参数梯度. 激活以固定的步进时间 $T_{bptt}$ 向前传播. 为此子序列计算交叉熵梯度, 并将其反向传播至其起点. 为了提高计算效率, 每个线程一次对四个发话的子序列进行操作, 因此矩阵乘法可以一次对四个帧进行并行操作. 我们使用异步随机梯度下降 (ASGD) 来优化网络参数, 从多核计算机上的多个线程异步更新参数. 这有效地增加了批次大小, 并减少了给定批次中帧的相关性. 线程更新了参数后, 它将在每种发音中继续执行下一个子序列, 以保留 LSTM 状态, 或者在完成时以重置状态开始新的发音. 请注意, 每个发音的最后一个子序列可以比 $T_{bptt}$ 短, 但会填充到全长, 尽管这些填充帧不会生成任何梯度. 

这种高度并行的单机 ASGD 框架对于我们用于带 DNN (数百万个参数) 的大规模 ASR 的规模训练模型证明是缓慢的. 为了进一步扩展, 我们在许多 (例如 500 台) 独立的机器上复制单机工作程序, 每台机器具有三个同步的计算线程. 每个工作人员与共享的分布式参数服务器通信, 该服务器存储 LSTM 参数. 当工作人员在一个微型批处理 (3 个 4 $T_{bptt}$ 帧) 上计算出参数梯度后, 梯度向量将被分割并发送到参数服务器分片, 每个分片将梯度添加到其参数并以新参数做出响应. 参数服务器分片聚合参数更新完全异步. 例如, 来自工作机的梯度更新可能以不同的顺序到达参数服务器的不同分片. 尽管存在异步性, 但我们观察到稳定的收敛, 尽管必须降低学习率, 这是可以预期的, 因为更大的并行度会增加有效的批处理大小. 



























