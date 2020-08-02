## 卷积神经网络的句子分类

https://arxiv.org/pdf/1408.5882.pdf



https://code.google.com/archive/p/word2vec/

https://blog.csdn.net/leyounger/article/details/79343404

### 摘要

我提报告了一系列在卷积神经网络 (CNN) 上进行的实验, 这些卷积神经网络在针对句子级别分类任务的预训练词向量的顶部进行了训练. 我们表明, 几乎没有超参数调整和静态矢量的简单 CNN 在多个基准上均能获得出色的结果. 通过微调学习特定于任务的向量可以进一步提高性能. 我们另外建议对体系结构进行简单修改, 以允许同时使用特定于任务的向量和静态向量. 本文讨论的 CNN 模型在 7 个任务中的 4 个改进了现有技术, 其中包括情感分析和问题分类. 





### 1. 介绍

近年来, 深度学习模型在计算机视觉和语音识别中取得了显著成果. 在自然语言处理中, 许多深度学习方法的工作涉及通过神经语言模型学习单词向量表示, 并在整个语言过程中进行合成, 通过学习的词向量进行分类. 词向量实质上是特征提取器, 其将词从稀疏的 1-V 编码 (这里 V 是词汇量) 通过隐藏层投影到低维向量空间上, 该特征提取器对词在其维度上的语义特征进行编码. 在这种密集表示中, 语义上接近的词在较低维向量空间中同样接近 (以欧氏距离或余弦距离). 卷积神经网络 (CNN) 利用带有卷积滤波器的图层应用于局部特征. CNN 模型最初是为计算机视觉而发明的, 后来被证明对 NLP 有效, 并且在语义解析, 搜索查询检索, 句子建模和其它传统 NLP 任务方面取得了优异的成绩. 

在目前的工作中, 我们训练了一个简单的 CNN, 在从无监督的神经语言模型获得的单词向量之上, 具有一层卷积. 这些向量由 Mikolov 等人在 1000 亿字的 Google 新闻中训练而来, 且是公开可用的. 最初, 我们使单词向量保持静态, 仅学习模型的其他参数. 尽管对超参数的调整很少, 但这个简单的模型在多个基准上均能获得出色的结果, 这表明预训练的向量是 "通用" 特征提取器, 可用于各种分类任务. 通过微调学习特定于任务的向量可进一步改进. 最后, 我们描述了对体系结构的简单修改, 以允许通过具有多个通道使用预训练向量和特定于任务的向量. 

我们的工作在哲学上与 Razavian 等人相似. 对于图像分类, 从预训练的深度学习模型中获取的特征提取器在各种任务上表现良好, 包括与训练特征提取器在原始任务截然不同的任务. 



### 2. 模型

如图 1 所示, 模型架构是 Collobert 等人的 CNN 架构的细微变化. 令 $\mathbf{x}_{i} \in \mathbb{R}^{k}$ 为对应于句子中第 $i$ 个词的 $k$ 维词向量. 长度为 $n$ 的句子 (必要时 padded) 表示为. 

$$\begin{aligned} \mathbf{x}_{1:n} = \mathbf{x}_{1} \oplus \mathbf{x}_{2} \oplus \cdots \oplus \mathbf{x}_{n}, \quad (1) \end{aligned}$$ 

其中: $\oplus$ 是级联操作. 通常, 让 $\mathbf{x}_{i: i+j}$ 表示单词 $\mathbf{x}_{i}, \mathbf{x}_{i+1}, \cdots , \mathbf{x}_{i+j}$ 的串联; 卷积运算涉及一个滤波器 $\mathbf{w} \in \mathbb{R}^{hk}$, 它应用于 $h$ 个字的窗口以产生新特征. 例如, 从单词 $\mathbf{x}_{i:i+h-1}$ 的窗口生成特征 $c_{i}$

$$\begin{aligned} c_{i} = f(\mathbf{w} \cdot \mathbf{x}_{i:i+h-1} + b). \quad (2) \end{aligned}$$ 

其中: $b \in \mathbb{R}$ 为偏置项, $f$ 是一个非线性函数, 例如双曲正切. 该滤波器应用于句子 $\{ \mathbf{x}_{1:h}, \mathbf{x}_{1:h+1},\cdots , \mathbf{x}_{n-h+1:n} \}$ 中的每个可能的单词窗品, 以生成特征图. 

$$\begin{aligned} \mathbf{c} = [c_{1}, c_{2}, \cdots, c_{n-h+1}], \quad (3) \end{aligned}$$ 

其中: $\mathbf{c} \in \mathbb{R}^{n-h+1}$. 然后, 我们在特征图上应用最大超时池化操作, 并将最大值 $\hat{c} = \max{\{\mathbf{c} \}}$ 作为与该特定过滤器相对应的特征. 这个想法是为每个特征图捕获最重要的特征 - 具有最大价值的特征. 这种池化方案自然可以处理可变的句子长度. 

我们已经描述了从一个过滤器中提取一个特征的过程. 该模型使用多个过滤器 (窗口大小不同) 来获取多个特征. 这些特征形成倒数第二层, 并传递到完全连接的 $softmax$ 层, 其输出是标签上的概率分布. 

在其中一种模型变体中, 我们尝试使用两个 "通道" 的单词向量, 其中一个在整个训练过程中保持不变, 而另一个通过反向传播进行微调. 在图 1 所示的多通道体系结构中, 每个滤波器都应用于两个通道, 并且将结果相加以计算公式 (2) 中的 $c_{i} $. 该模型在其他方面等效于单通道体系结构. 



### 2.1 正则化

为了进行正则化, 我们在倒数第二层采用了权重向量的 l2 范数约束. $dropout$ 通过在正向反向传播期间随机删除隐藏单元的比例 $p$ (即设置为零) 来防止隐藏单元的共适应. 也就是说, 给定倒数第二层 $\mathbf{z} = [\hat{c}_{1}, \cdots , \hat{c}_{m}]$ (请注意, 这里有 $m$ 个过滤器), 而不是使用:  

$$\begin{aligned} y = \mathbf{w} \cdot \mathbf{z} + b \quad (4) \end{aligned}$$ 

对于正向传播中的输出单元 $y$, $dropout$ 使用

$$\begin{aligned} y = \mathbf{w} \cdot (\mathbf{z} \circ \mathbf{r}) + b, \quad (4) \end{aligned}$$ 

其中: $\circ$ 是逐元素乘法运算符, $\mathbf{r} \in \mathbb{R}^{m}$ 是蒙版向量, 为 Bernoulli 随机变量, 为 1 的概就为 $p$. 梯度只通过未被掩盖的单元传播. 在测试时, 将学习的权重向量按 $p$ 进行缩放, 如 $\hat{\mathbf{w}} = p \mathbf{w}$, 并且使用 $\hat{\mathbf{w}}$ (未进行 $dropout$)来对未看见的句子评分. 我们另外通过重新缩放 $\mathbf{w}$ 来约束权重向量的 $l2$-范数, 以使得在梯度下降时, 如果 ${||\mathbf{w}||}_{2} > s$ 则将其缩放到 ${||\mathbf{w}||}_{2} = s$ . 





练习实例: (不收敛)

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, x, y_true, y_pred):
        self.x = x
        self.y_true = y_true
        self.y_pred = y_pred

        self.loss = None
        self.metrics = None
        self.optimizer = None

        self.sess = None

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def fit(self, x, y, batch_size, epochs):
        loss = self.loss(self.y_true, self.y_pred)
        metrics = list()
        for metric_fn in self.metrics:
            metric = metric_fn(self.y_true, self.y_pred)
            metrics.append(metric)

        train_op = self.optimizer.minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = list()
            epoch_metrics = list()
            for batch_x, batch_y in self.get_next_batch(x, y, batch_size=batch_size):
                _, loss_, metrics_ = self.sess.run(
                    fetches=[train_op, loss, metrics],
                    feed_dict={self.x: batch_x, self.y_true: batch_y}
                )

                epoch_loss.append(loss_)
                epoch_metrics.append(metrics_)

            loss_ = np.mean(np.array(epoch_loss), axis=0)
            metrics_ = np.mean(np.array(epoch_metrics), axis=0)
            print(f'epoch: {epoch}, loss: {loss_}, metrics: {metrics_}')
        return

    def predict(self, inputs, model_path):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, model_path)
        output = self.sess.run(self.y_pred, feed_dict={self.x: inputs})
        self.sess.close()
        return output

    def save(self, save_path='models'):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
        return

    @staticmethod
    def get_next_batch(x, y, batch_size):
        l = len(x)
        idx = np.arange(l)
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]
        steps = l // batch_size

        for step in range(steps):
            b_idx = step * batch_size
            e_idx = b_idx + batch_size
            batch_x = x[b_idx: e_idx]
            batch_y = y[b_idx: e_idx]
            yield batch_x, batch_y

    @staticmethod
    def loss(y_true, y_pred, l2_loss=None, l2_reg_lambda=0.0):
        if l2_loss is None:
            l2_loss = 0.0
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
            losses = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        return losses

    @staticmethod
    def accuracy(y_true, y_pred):
        with tf.name_scope('accuracy'):
            y_pred = tf.nn.softmax(logits=y_pred, axis=-1, name='softmax')
            y_pred = tf.argmax(y_pred, axis=-1, name='argmax')
            y_true = tf.argmax(y_true, axis=-1)
            correct = tf.equal(y_pred, y_true)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        return accuracy


class TextConv(object):
    """
    VALID: 卷积, 只在卷积核 kernel 始终在 input 内, 不能 padding 到外部. 所以卷积后的 output 大小比 input 要小.
    SAME: 卷积, 通过 padding, 卷积核可以扩展到 input 外部, 以使得卷积后的 output 大小与 input 相同.
    """
    def __init__(self, height, width, in_channels, out_channels, variable_scope=None):
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable_scope = 'text_conv' if variable_scope is None else variable_scope

    def __call__(self, input):
        if isinstance(self.variable_scope, str):
            scope = tf.variable_scope(self.variable_scope)
        else:
            scope = self.variable_scope
        with scope:
            return self.text_conv(input)

    def text_conv(self, input):
        w = tf.get_variable(
            name='w',
            shape=(self.height, self.width, self.in_channels, self.out_channels),
            initializer=tf.truncated_normal_initializer(stddev=0.1),
        )

        b = tf.get_variable(
            name='b',
            shape=(self.out_channels,),
            initializer=tf.constant_initializer(value=0.1),
        )

        # shape = (batch, seq_len, 1, out_channels)
        features = tf.nn.conv2d(input=input, filter=w, strides=(1, 1, 1, 1), padding='VALID', name='conv2d')
        features = tf.nn.bias_add(features, b, name='bias_add')
        features = tf.nn.relu(features=features, name='relu')

        seq_len = input.shape[1]

        # 只在 seq_len 方向进行宽度为 1 的最大池化.
        # shape = (batch, 1, 1, out_channels)
        features = tf.nn.max_pool(
            value=features,
            ksize=(1, seq_len - self.height + 1, 1, 1),
            strides=(1, 1, 1, 1),
            padding='VALID',
            name='pool'
        )
        return features


class TextCNN(object):
    def __init__(self, seq_len, embedding_size, n_classes, filter_sizes: list, n_filters: int, dropout_keep_prob, l2_reg_lambda=0.0):
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda

        self.text_conv_list = self.init_text_conv()

    def init_text_conv(self):
        kernels = list()
        for filter_size in self.filter_sizes:
            kernel = TextConv(
                height=filter_size,
                width=self.embedding_size,
                in_channels=1,
                out_channels=self.n_filters,
            )
            kernels.append(kernel)
        return kernels

    def build(self):
        x = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.embedding_size, 1], name="input_x")
        y_true = tf.placeholder(tf.float32, shape=[None, self.n_classes], name="input_y")

        o1 = self.conv_on_inputs(x)
        y_pred, _ = self.ffnn(o1)

        model = Model(x, y_true, y_pred)
        return model

    def conv_on_inputs(self, inputs):
        with tf.name_scope('conv_on_input'):
            output_list = list()
            for i, text_conv in enumerate(self.text_conv_list):
                with tf.variable_scope(f'text_conv_{i}'):
                    output = text_conv(inputs)
                    output_list.append(output)
            s = self.n_filters * len(self.filter_sizes)
            feature = tf.concat(output_list, -1, name='concat')
            feature = tf.reshape(feature, shape=(-1, s))
            feature = tf.nn.dropout(feature, keep_prob=self.dropout_keep_prob, name='dropout')
        return feature

    def ffnn(self, inputs):
        in_dims = self.n_filters * len(self.filter_sizes)

        with tf.name_scope('ffnn'):
            w = tf.get_variable(
                name='w',
                shape=(in_dims, self.n_classes),
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )

            b = tf.get_variable(
                name='b',
                shape=(self.n_classes,),
                initializer=tf.constant_initializer(value=0.1)
            )

            l2_loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)

            output = tf.nn.xw_plus_b(inputs, w, b, name='scores')
        return output, l2_loss


def demo1():
    def load_data(n1=10000, n2=10000):
        inputs1 = np.random.randn(n1, 10, 3, 1) + 50
        labels1 = np.array([[1, 0]], dtype=np.float32)
        labels1 = np.tile(labels1, (n1, 1))

        inputs2 = np.random.randn(n2, 10, 3, 1) - 50
        labels2 = np.array([[0, 1]], dtype=np.float32)
        labels2 = np.tile(labels2, (n2, 1))

        inputs = np.concatenate([inputs1, inputs2])
        labels = np.concatenate([labels1, labels2])
        return inputs, labels

    inputs, labels = load_data()

    model = TextCNN(
        seq_len=10,
        embedding_size=3,
        n_classes=2,
        filter_sizes=[2, 3],
        n_filters=2,
        dropout_keep_prob=0.9,
        l2_reg_lambda=0.0
    ).build()

    optimizer = tf.train.AdamOptimizer()
    loss = Model.loss
    accuracy = Model.accuracy

    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

    model.fit(inputs, labels, batch_size=50, epochs=10)
    return


if __name__ == '__main__':
    demo1()

```



























