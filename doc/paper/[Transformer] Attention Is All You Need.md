## Attention Is All You Need

https://arxiv.org/pdf/1706.03762.pdf

https://github.com/yuenoble/Machine-Translation-by-Transformer

https://github.com/DongjunLee/transformer-tensorflow

### 摘要

好的序列模型基于复杂的递归或卷积神经网络, 包括编码器和解码器. 表现最佳的模型还通过注意力机制连接编码器和解码器. 我们提出了一种新的简单网络架构, 即 Transformer, 它完全基于注意力机制, 完全消除了重复和卷积. 在两个机器翻译任务上进行的实验表明, 这些模型在质量上具有优势, 同时具有更高的可并行性, 并且所需的训练时间明显更少. 我们的模型在 WMT 2014 英德翻译任务中达到 28.4 BLEU, 比包括 ensembles 在内的现有最佳结果提高了 2 BLEU. 在 WMT 2014 英语到法语翻译任务中, 我们的模型在八个 GPU 上进行了 3.5 天的训练后, 建立了新的单模型最新 BLEU 分数 41.8, 这是最佳训练成本的一小部分文献中的模型. 我们展示了 Transformer 通过将其成功地应用于具有大量训练数据和有限训练数据的英语选区解析, 可以很好地概括其他任务. 



### 1 介绍

递归神经网络, 特别是长短期记忆和 gated 递归神经网络已被牢固地确立为序列建模和转导问题 (例如语言建模和机器翻译) 中的最新方法. 此后, 人们一直在不断努力以扩大循环语言模型和编码器-解码器体系结构的边界. 

循环模型通常沿输入和输出序列的符号位置考虑计算. 将位置与计算时间的步骤对齐, 它们会根据先前的隐藏状态 $h_{t-1}$ 和位置 $t$ 的输入生成一系列隐藏状态 $h_{t}$. 这种固有的顺序性质阻止了训练样本内的并行化, 这在较长的序列长度上变得至关重要, 因为内存限制限制了示例之间的批处理. 最近的工作通过分解技巧和条件计算在计算效率上取得了显著提高, 同时在后者的情况下还提高了模型性能. 但是, 顺序计算的基本约束仍然存在. 

注意机制已成为各种任务中引人注目的序列建模和转导模型不可或缺的一部分, 从而允许对依赖项进行建模而无需考虑它们之间的距离输入或输出序列. 除了少数情况, 此类注意力机制都与循环网络结合使用. 

在这项工作中, 我们提出了 Transformer, 它是一种避免重复发生的模型体系结构, 而是完全依赖于注意力机制来绘制输入和输出之间的全局依存关系. 在 8 个 P100 GPU 上进行了长达 12 个小时的训练之后, 该 Transformer 可以实现更多的并行化, 并且可以在翻译质量方面达到新的水平. 



### 2 背景

减少顺序计算的目标也构成了扩展神经 GPU, ByteNet 和 ConvS2S 基础, 它们全部使用卷积神经网络作为基本构件, 并行计算所有输入和输出位置的隐藏表示. 在这些模型中, 关联来自两个任意输入或输出位置的信号所需的操作数在位置之间的距离中增加, 对于 ConvS2S 线性增长, 而对于 ByteNet 则对数增长. 这使得学习远处位置之间的依存关系变得更加困难. 在 Transformer 中, 此操作被减少为恒定的操作次数, 尽管以平均注意力加权位置为代价, 但是我们用第 3.2 节中所述的多头注意力抵消了这种影响. 

Self-attention, 有时也称为 intra-attention, 是一种与单个序列的不同位置相关的注意力机制, 目的是计算序列的表示形式. self-attention 已成功用于各种任务中, 包括阅读理解, 抽象总结, 文本蕴涵和学习与任务无关的句子表示. 

端到端内存网络基于递归注意机制, 而不是序列对齐的递归, 并且已被证明在简单语言问题和语言建模任务中表现良好. 

据我们所知, Transformer 是第一个完全依靠自我注意力来计算其输入和输出的转导模型, 而无需使用序列对齐的 RNN 或卷积. 在以下各节中, 我们将描述 Transformer, motivate, self-attention, 并讨论其相对优势. 



### 3 模型结构

大多数竞争性神经序列转导模型都具有编码器-解码器结构. 在此, 编码器映射符号表示的输入序列 $(x_{1}, \cdots , x_{n})$ 到一个连续值的序列表示 $\mathbf{z} = (z_{1}, \cdots , z_{n})$. 给定 $\mathbf{z}$, 然后, 解码器一次生成一个符号的输出序列 $(y_{1}, \cdots , y_{m})$. 模型的每一步都是自回归的, 在生成下一个时, 会将先前生成的符号用作附加输入. 

Transformer 遵循这种总体架构, 对编码器和解码器使用堆叠式自注意力和逐点, 全连接层, 全别如图 1 的左半部和和右半部分所示. 



### 3.1 编码和解码栈

**编码:** 编码器由 $N=6$ 个相同层的堆叠组成. 每层都有两个子层. 第一个是 **multi-head self-attention** 机制, 第二个是简单的位置完全连接的前馈网络. 我们在两个子层的每一层采用残差连接, 然后进行层归一化. 也就是说, 每个子层的输出是 $LayerNorm(x + Sublayer(x))$, 其中 $Sublayer(x)$ 是由子层本身实现的功能. 为了方便这些残差连接, 模型中的所有子层以及嵌入层均产生尺寸为 $d_{model} = 512$ 的输出. 

**解码:** 解码器也由 $N=6$ 个相同层堆叠组成. 除了每个编码器层中的两个子层之外, 解码器还插入第三子层, 该第三子层对编码器堆栈的输出执行 **multi-head attention**. 与编码器类似, 我们在每个子层周围采用残差连接, 然后进行层归一化. 我们还修改了解码器堆叠中的 **self-attention** 子层, 以防止位置出现在后续位置. 这种掩盖结合输出嵌入被一个位置偏移的事实, 确保了对位置 $i$ 的预测只能依赖小于 $i$ 位置的已知输出. 



### 3.2 注意

一个注意函数可以描述为将查询和一组 key-value 对映射到输出, 其中 query, key, value 和输出都是向量. 将输出计算为值的加权总和, 其中分配给每个值的权重是通过查询与相应键的兼容性函数来计算的. 



### 3.2.1 Scaled Dot-Product Attention

我们称我们的特别注意为 "Scaled Dot-Product Attention" (图 2). 输入由维数为 $d_{k}$ 的查询和键以及维数为 $d_{v}$ 的值组成. 我们使用所有键计算查询的点积, 将每个键除以 $\sqrt{d_{k}}$, 然后应用 $softmax$ 函数获得值的权重. 

实际上, 我们在一组查询上同时计算注意力函数, 并将它们打包成矩阵 $Q$. 键和值也打包到矩阵 $K$ 和 $V$ 中. 我们将输出矩阵计算为: 

$$\begin{aligned} Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V \quad (1) \end{aligned}$$ 

两个最常用的注意力函数是加性注意力和点积 (乘法) 注意力. 除了比例因子 $\sqrt{d_{k}}$ 之外, 点积注意与我们的算法相同. 加性注意事项使用具有单个隐藏层的前馈网络来计算兼容性函数. 尽管两者在理论上的复杂度相似, 但是在实践中, 点积的关注要快得多, 并且空间效率更高, 因为可以使用高度优化的矩阵乘法代码来实现. 

虽然对于较小的 $d_{k}$ 而言, 两种机制的表现相似, 但加性注意的效果优于点积的注意, 而对于较大的 $d_{k}$ 则不进行缩放. 我们怀疑对于较大的 $d_{k}$ 值, 点积会增大幅度, 从而将 $softmax$ 函数堆入梯度极小的区域. 为了抵消这种影响, 我们通过 $\frac{1}{\sqrt{d_{k}}}$. 



```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


def tf_print(tensor):
    print(sess.run(tensor))

    
def scaled_dot_product(qs, ks, vs, masked=True):
    dk = tf.cast(tf.shape(qs)[-1], 'float32')
    o1 = tf.matmul(qs, ks, transpose_b=True)
    # (batch_size, num_heads, query_dim, key_dim)
    o2 = o1 / (dk ** 0.5)

    if masked:
        # (query_dim, key_dim)
        diag_vals = tf.ones_like(o2[0, 0, :, :])
        # (q_dim, k_dim)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.tile(
            tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
            [tf.shape(o2)[0], tf.shape(o2)[1], 1, 1]
        )
        paddings = tf.ones_like(masks) * -1e9
        tf_print(masks)
        o2 = tf.where(tf.equal(masks, 0), paddings, o2)

    o3 = tf.nn.softmax(o2)
    return tf.matmul(o3, vs)


def scaled_dot_product_v2(qs, ks, vs, masked=True):
    dk = tf.cast(tf.shape(qs)[-1], 'float32')
    o1 = tf.matmul(qs, ks, transpose_b=True)
    # (batch_size, query_dim, key_dim)
    o2 = o1 / (dk ** 0.5)

    if masked:
        # (query_dim, key_dim)
        diag_vals = tf.ones_like(o2[0, :, :])
        # (q_dim, k_dim)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()

        masks = tf.tile(
            tf.reshape(tril, [1] + tril.get_shape().as_list()),
            [tf.shape(o2)[0], 1, 1]
        )
        paddings = tf.ones_like(masks) * -1e9
        tf_print(masks)
        o2 = tf.where(tf.equal(masks, 0), paddings, o2)

    o3 = tf.nn.softmax(o2)
    return tf.matmul(o3, vs)


qs = tf.constant(
    value=[[1, 2, 3],
           [3, 2, 1],
           [2, 3, 4],
           [4, 3, 2]],
    dtype=tf.float32
)

qs = tf.expand_dims(qs, axis=0)
qs = tf.expand_dims(qs, axis=0)

with tf.Session() as sess:
    product = scaled_dot_product(qs, qs, qs)
    ret = sess.run(product)
    print(ret)

```

以上代码输出结果为: 

```text
# scaled_dot_product
[[[[1. 0. 0. 0.]
   [1. 1. 0. 0.]
   [1. 1. 1. 0.]
   [1. 1. 1. 1.]]]]
[[[[1.        2.        3.       ]
   [2.819305  1.9999999 1.1806947]
   [1.9950423 2.993949  3.9928553]
   [3.8137977 2.9944925 2.175187 ]]]]
(1, 1, 4, 3)

# scaled_dot_product_v2
[[[1. 0. 0. 0.]
  [1. 1. 0. 0.]
  [1. 1. 1. 0.]
  [1. 1. 1. 1.]]]
[[[1.        2.        3.       ]
  [2.819305  1.9999999 1.1806947]
  [1.9950423 2.993949  3.9928553]
  [3.8137977 2.9944925 2.175187 ]]]
```









### 3.2.2 Multi-Head Attention

与其使用 $d_{model}$ 维键, 值和查询执行单个关注函数, 我们发现将查询, 键和值线性投影不同的 $h$ 次是有益的, 分别学习到 $d_{k}$, $d_{k}$ 和 $d_{v}$ 尺寸的线性投影. 然后, 在查询, 键和值每个这些预计的版本上, 我们并行执行注意力函数, 从而意生 $d_{v}$ 维输出值. 将它们连接起来并再次投影, 得到最终值, 如图 2 所示. 

多头注意力使模型可以共同关注来自不同位置的不同表示子空间的信息. 对于一个注意力头, 平均会抑制这种情况. 

$$\begin{aligned} MultiHead(Q,K,V) &= Concat(head_{1}, \cdots , head_{h})W^{O} \\ \text{其中: } head_{i} &= Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V}) \end{aligned}$$

其中映射矩阵为: $W_{i}^{Q} \in \mathbb{R}^{d_{model} \times d_{k}}$, $W_{i}^{K} \in \mathbb{R}^{d_{model} \times d_{k}}$, $W_{i}^{V} \in \mathbb{R}^{d_{model} \times d_{v}}$, $W_{i}^{O} \in \mathbb{R}^{hd_{v} \times d_{model}}$. 

在这项工作中, 我们使用 $h=8$ 的平行注意层或头. 对于每一个, 我们使用 $d_{k} = d_{v} = d_{model}/h = 64$. 由于每个头部的尺寸减少, 因此总的计算成本类似于具有全尺寸的单头注意力的计算成本. 



练习实例: 论文中 Multi-Head Attention 的实现. 

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


def linear_projection(q, k, v, dk, dv):
    # 映射矩阵. 
    q = tf.layers.dense(q, dk, use_bias=False)
    k = tf.layers.dense(k, dk, use_bias=False)
    v = tf.layers.dense(v, dv, use_bias=False)
    return q, k, v


def scaled_dot_product_v2(qs, ks, vs, masked=True):
    dk = tf.cast(tf.shape(qs)[-1], 'float32')
    o1 = tf.matmul(qs, ks, transpose_b=True)
    o2 = o1 / (dk ** 0.5)
    o2_shape = tf.shape(o2).eval()
    o2_rank = len(o2_shape)

    if masked:
        diag_vals = tf.ones(o2_shape[-2:])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        n = o2_rank - 2
        masks = tf.tile(
            tf.reshape(tril, [1] * n + tril.get_shape().as_list()),
            [*o2_shape[:n], 1, 1]
        )
        paddings = tf.ones_like(masks) * -1e9
        o2 = tf.where(tf.equal(masks, 0), paddings, o2)
    o3 = tf.nn.softmax(o2)
    return tf.matmul(o3, vs)


def multi_heads(q, k, v, dk, dv):
    embedding_size = tf.shape(q)[-1]
    num_heads = embedding_size // dk
    head_list = []
    for i in range(num_heads.eval()):
        qs, ks, vs = linear_projection(q, k, v, dk, dv)
        head_i = scaled_dot_product_v2(qs, ks, vs, masked=True)
        head_list.append(head_i)
    heads = tf.concat(head_list, axis=-1)
    return heads


q = tf.constant(
    value=[[1, 2, 3],
           [3, 2, 1],
           [2, 3, 4],
           [4, 3, 2]],
    dtype=tf.float32
)

q = tf.expand_dims(q, axis=0)

with tf.Session() as sess:
    # 由于 multi_heads 中有 `num_heads.eval()`, 所以此图必须要在 sess 里. 
    heads = multi_heads(q, q, q, 1, 1)

    sess.run(tf.global_variables_initializer())
    ret = sess.run(heads)
    print(ret)
    print(ret.shape)

```



以上代码输出结果为: 

```text
[[[-3.612752  -1.4414054  1.5105194]
  [-3.5357485 -1.3180245  1.7728679]
  [-5.3133264 -1.502743   2.4817638]
  [-5.266579  -1.5836629  1.8639469]]]
(1, 4, 3)
```



练习实例: 此实现直接将原矩阵 split. 不是论文中的实现. 

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


def tf_print(tensor):
    print(sess.run(tensor))


def split_heads(q, k, v, dk, dv):
    def split_last_dimension_then_transpose(tensor, d):
        t_shape = tensor.get_shape().as_list()
        embedding_size = t_shape[-1]

        # (batch_size, max_seq_len, num_heads, dim)
        new_shape = t_shape[:-1] + [embedding_size // d, d]
        tensor = tf.reshape(tensor, new_shape)

        # (batch_size, num_heads, max_seq_len, dim)
        ret = tf.transpose(tensor, [0, 2, 1, 3])
        return ret

    qs = split_last_dimension_then_transpose(q, dk)
    ks = split_last_dimension_then_transpose(k, dk)
    vs = split_last_dimension_then_transpose(v, dv)
    return qs, ks, vs


q = tf.constant(
    value=[[1, 2, 3],
           [3, 2, 1],
           [2, 3, 4],
           [4, 3, 2]],
    dtype=tf.float32
)

q = tf.expand_dims(q, axis=0)


with tf.Session() as sess:
    heads = split_heads(q, q, q, 1, 1)
    qs, ks, vs = sess.run(heads)
    print(qs)
    print(ks)
    print(vs)
    print(qs.shape)

```



以上代码输出结果为: 

```text
[[[[1.]
   [3.]
   [2.]
   [4.]]

  [[2.]
   [2.]
   [3.]
   [3.]]

  [[3.]
   [1.]
   [4.]
   [2.]]]]
   
[[[[1.]
   [3.]
   [2.]
   [4.]]

  [[2.]
   [2.]
   [3.]
   [3.]]

  [[3.]
   [1.]
   [4.]
   [2.]]]]
   
[[[[1.]
   [3.]
   [2.]
   [4.]]

  [[2.]
   [2.]
   [3.]
   [3.]]

  [[3.]
   [1.]
   [4.]
   [2.]]]]
(1, 3, 4, 1)
```



### Attention 实现

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class AttentionHelper(object):

    @classmethod
    def get_lower_triangle_mask(cls, to_mask_shape):
        """
        :param to_mask_shape: tuple or list. 指示所生成的 mask, 的目标形状.
        :return:
        """
        rank = len(to_mask_shape)
        diag_vals = tf.ones(to_mask_shape[-2:])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        n = rank - 2
        lower_tri_m = tf.tile(
            tf.reshape(tril, [1] * n + tril.get_shape().as_list()),
            [*to_mask_shape[:n], 1, 1]
        )
        return lower_tri_m

    @classmethod
    def get_upper_triangle_mask(cls, to_mask_shape):
        lower_tri_m = cls.get_lower_triangle_mask(to_mask_shape)
        rank = len(to_mask_shape)
        t = np.arange(rank)
        upper_tri_m = tf.transpose(lower_tri_m, [*t[:-2], t[-2], t[-1]])
        return upper_tri_m

    @classmethod
    def merge_mask1_and_mask2(cls, mask1, mask2):
        t_dtype = mask1.dtype
        mask1 = tf.cast(tf.not_equal(mask1, 0), tf.float32)
        mask2 = tf.cast(tf.not_equal(mask2, 0), tf.float32)
        mask_bool = tf.cast(tf.equal(mask1 + mask2, 2), dtype=tf.bool)
        mask = tf.cast(mask_bool, dtype=t_dtype)
        return mask

    @classmethod
    def apply_mask(cls, tensor, mask, value):
        paddings = tf.ones_like(mask) * value
        result = tf.where(tf.equal(mask, 0), paddings, tensor)
        return result

    @classmethod
    def expand_mask(cls, mask, dim):
        rank = tf.rank(mask)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1] * rank.eval() + [dim])
        return mask


class Attention(AttentionHelper):
    L_TRI_MASK = 'lower_triangle_mask'
    U_TRI_MASK = 'upper_triangle_mask'
    NO_MASK = 'none'

    def __init__(self, num_heads=1, triangle_mask=None, dropout=0.1):
        """
        :param num_heads: scalar.
        :param triangle_mask: str. choices:
        {`lower_triangle_mask`, `upper_triangle_mask`, `none`}.
        """
        self.num_heads = num_heads
        if triangle_mask not in {
            self.L_TRI_MASK,
            self.U_TRI_MASK,
            self.NO_MASK,
            None
        }:
            raise ValueError(
                f'triangle mask parameter must be one of: '
                f'`{self.L_TRI_MASK}`, `{self.U_TRI_MASK}`, `{self.NO_MASK}`. '
            )
        self._triangle_mask = triangle_mask if triangle_mask is not None else self.NO_MASK
        self._dropout = dropout

        self.embedding_size = None
        # sequence 张量的类型.
        self._t_type = None
        self._seq_shape = None
        self._seq_rank = None

    def __call__(self, q, k, v, q_mask=None, k_mask=None):
        self._t_type = q.dtype
        self._seq_shape = tf.shape(q).eval()
        self._seq_rank = len(self._seq_shape)
        self.embedding_size = self._seq_shape[-1]

        if q_mask is None:
            q_mask = tf.constant(
                value=1, shape=self._seq_shape[:-1],
                dtype=self._t_type
            )
        if k_mask is None:
            k_mask = tf.constant(
                value=1, shape=self._seq_shape[:-1],
                dtype=self._t_type
            )

        if self.embedding_size % self.num_heads != 0:
            raise ValueError(
                f'the embedding-size must be multiple of num-of-heads, '
                f'however: embeding size: '
                f'{self.embedding_size}, num-of-heads: {self.num_heads}'
            )

        result = self._multi_heads(q, k, v, q_mask, k_mask)
        return result

    def _multi_heads(self, q, k, v, q_mask=None, k_mask=None):
        q_mask = tf.cast(q_mask, dtype=q.dtype)
        k_mask = tf.cast(k_mask, dtype=q.dtype)

        emd_q = tf.shape(q)[-1]
        emd_v = tf.shape(v)[-1]

        dk = (emd_q // self.num_heads).eval()
        dv = (emd_v // self.num_heads).eval()

        q_mask = self.expand_mask(q_mask, dk)
        k_mask = self.expand_mask(k_mask, dv)

        head_list = []
        if self.num_heads > 1:
            for i in range(self.num_heads):
                qs, ks, vs = self._linear_projection(q, k, v, dk, dv)
                head_i = self._scaled_dot_product(qs, ks, vs, q_mask, k_mask)
                head_list.append(head_i)
            heads = tf.concat(head_list, axis=-1)
        elif self.num_heads == 1:
            heads = self._scaled_dot_product(q, k, v, q_mask, k_mask)
        else:
            raise ValueError(
                f'num-of-heads: '
                f'{self.num_heads} should not less than 1. '
            )
        result = tf.nn.dropout(heads, keep_prob=1.0 - self._dropout)
        return result

    @staticmethod
    def _linear_projection(q, k, v, dk, dv):
        q = tf.layers.dense(q, dk, use_bias=False)
        k = tf.layers.dense(k, dk, use_bias=False)
        v = tf.layers.dense(v, dv, use_bias=False)
        return q, k, v

    def _scaled_dot_product(self, q, k, v, q_mask, k_mask):
        """
        同时兼容 qs, ks, vs 形状为:
        (batch, sequence_len, embedding_size)
        或:
        (batch, num_heads, sequence_len, embedding_size)
        """
        dk = tf.cast(tf.shape(q)[-1], self._t_type)
        o1 = tf.matmul(q, k, transpose_b=True)
        o2 = o1 / (dk ** 0.5)

        o2_shape = tf.shape(o2).eval()
        co_seq_m = tf.matmul(q_mask, k_mask, transpose_b=True)

        if self._triangle_mask == self.L_TRI_MASK:
            lower_tri_m = self.get_lower_triangle_mask(o2_shape)
            mask = self.merge_mask1_and_mask2(co_seq_m, lower_tri_m)
        elif self._triangle_mask == self.U_TRI_MASK:
            upper_tri_m = self.get_upper_triangle_mask(o2_shape)
            mask = self.merge_mask1_and_mask2(co_seq_m, upper_tri_m)
        else:
            mask_bool = tf.not_equal(co_seq_m, 0)
            mask = tf.cast(mask_bool, dtype=self._t_type)
            # for i in range(len(o2_shape) - 2):
            #     mask = tf.expand_dims(mask, axis=0)

        o2 = self.apply_mask(o2, mask, -1e9)
        o3 = tf.nn.softmax(o2)
        output = tf.matmul(o3, v) * q_mask
        return output


q = tf.constant(
    value=[[1, 2, 3, 1],
           [3, 2, 1, 3],
           [3, 2, 1, 3],
           [3, 2, 1, 3],
           [4, 3, 2, 4],
           [0, 0, 0, 0],
           [0, 0, 0, 0]],
    dtype=tf.float32
)

mask = tf.constant([1, 1, 1, 1, 1, 0, 0])

q = tf.expand_dims(q, axis=0)
mask = tf.expand_dims(mask, axis=0)

q = tf.expand_dims(q, axis=0)
mask = tf.expand_dims(mask, axis=0)


with tf.Session() as sess:
    # 由于 multi_heads 中有 `num_heads.eval()`, 所以此图必须要在 sess 里.
    attention = Attention(
        num_heads=2,
        triangle_mask=Attention.NO_MASK
    )
    heads = attention(q, q, q, mask, mask)
    # heads = attention(q, q, q)
    sess.run(tf.global_variables_initializer())
    ret = sess.run(heads)
    print(ret)
    print(ret.shape)

```



以上代码输出结果为: 

```text
# heads = attention(q, q, q, mask, mask)
[[[[ 3.3199599  5.572652  -4.8456287 -4.620596 ]
   [ 3.2848067  5.5240455 -4.867072  -4.633607 ]
   [ 3.2848067  5.5240455 -0.        -4.633607 ]
   [ 3.2848067  0.        -4.867072  -4.633607 ]
   [ 3.3080819  5.557107  -4.867253  -4.6337166]
   [ 0.         0.        -0.        -0.       ]
   [ 0.         0.        -0.        -0.       ]]]]
(1, 1, 7, 4)

# heads = attention(q, q, q)
[[[[ 0.67017984 -2.2749786  -1.4971381   4.6538124 ]
   [ 0.         -2.717022   -1.8176202   5.1857243 ]
   [ 0.8017726  -2.717022   -1.8176202   5.1857243 ]
   [ 0.8017726  -2.717022   -1.8176202   0.        ]
   [ 0.84689236 -2.90328    -1.8260543   5.303563  ]
   [ 0.49190134 -1.6282058  -0.7375144   2.9330318 ]
   [ 0.49190134 -1.6282058  -0.7375144   2.9330318 ]]]]
(1, 1, 7, 4)
```



















### 3.2.3 我们的模型中的注意应用 

Transformer 以三种不同方式使用 multi-head attention. 

* 在 "编码器-解码器注意" 层中, 查询来自先前的解码层, 而存储键和值来自编码器的输出. 这允许解码器中的每个位置都参与输入序列中的所有位置. 这模仿了序列到序列模型中的典型编码器-解码器注意机制. 
* 编码器包含自我注意层. 在自我关注层中, 所有键, 值和查询都来自同一位置, 在这种情况下, 是编码器中上一层的输出. 编码器中的每个位置都可以覆盖编码器上一层中的所有位置. 
* 类似地, 解码器中的自我注意层允许解码器上的每个位置关注直到并包括该位置的解码器上的所有位置. 我们需要防止解码器中向左流动信息, 以保留自回归属性. 通过屏蔽 (设置为 $- \infty$) softmax 输入中与非法连接相对应的所有值, 我们在扩展点乘积注决的内部实现了这一点. 参见图 2. 



### 3.3 前馈网络

除了关注子层之外, 我们的编码器和解码器中的每个层还包含一个完全连接的前馈网络, 该网络被分别并相同地应用于每个位置. 这由两个线性变换组成, 两个线性变换之间具有 ReLU 激活. 

$$\begin{aligned} FFN(x) = \max{(0, xW_{1} + b_{1})W_{2} + b_{2}} \quad (2) \end{aligned}$$ 

虽然线性变换在不同位置上相同, 但是它们使用不同的参数. 另一种描述方式是内核大小为 1 的两个卷积. 输入和输出的维数为 $d_{model} = 512$, 内层的维数为 $d_{ff} = 2048$. 



练习实例: 

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


class FFN(object):
    """FFN class (Position-wise Feed-Forward Networks)"""

    def __init__(self, ffn_dim=2048, model_dim=512, dropout=0.1):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout

    def __call__(self, inputs):
        # return self.dense_relu_dense(inputs)
        return self.conv_relu_conv(inputs)

    def dense_relu_dense(self, inputs):
        output = tf.layers.dense(inputs, self.ffn_dim, activation=tf.nn.relu)
        output = tf.layers.dense(output, self.model_dim)
        return tf.nn.dropout(output, 1.0 - self.dropout)

    def conv_relu_conv(self, inputs):
        outputs = tf.layers.conv1d(
            inputs, filters=self.ffn_dim, kernel_size=1, padding='SAME')
        outputs = tf.nn.relu(outputs)
        outputs = tf.layers.conv1d(
            outputs, filters=self.model_dim, kernel_size=1, padding='SAME')
        return outputs


inputs = tf.constant(
    value=[[1, 2, 3, 1],
           [3, 2, 1, 3],
           [3, 2, 1, 3],
           [3, 2, 1, 3],
           [4, 3, 2, 4],
           [0, 0, 0, 0],
           [0, 0, 0, 0]],
    dtype=tf.float32
)

inputs = tf.expand_dims(inputs, axis=0)

with tf.Session() as sess:
    ffn = FFN(ffn_dim=8, model_dim=4)
    outputs1 = ffn.dense_relu_dense(inputs)
    outputs2 = ffn.conv_relu_conv(inputs)

    sess.run(tf.global_variables_initializer())

    ret = sess.run(outputs1)
    print(ret)
    print(ret.shape)
    ret = sess.run(outputs2)
    print(ret)
    print(ret.shape)

```



以上代码输出结果为: 

```text
[[[ 0.          2.281394    0.61944866 -0.4328335 ]
  [ 0.22963138 -0.00837307 -0.21194808  0.14032087]
  [ 0.22963138 -0.00837307 -0.          0.14032087]
  [ 0.22963138 -0.00837307 -0.21194808  0.14032087]
  [ 0.4698934   0.          0.04317204  0.1971323 ]
  [ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]]]
(1, 7, 4)
[[[-0.50328743  2.2757394   0.8804463  -1.362143  ]
  [ 0.6775453   0.5728147  -0.02496564 -2.1623995 ]
  [ 0.6775453   0.5728147  -0.02496564 -2.1623995 ]
  [ 0.6775453   0.5728147  -0.02496564 -2.1623995 ]
  [ 0.73092574  1.2354627   0.12123942 -3.101633  ]
  [ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]]]
(1, 7, 4)
```









### 3.4 $Embeddings$ 和 $Softmax$ 

与其他序列转导模型相似, 我们使用学习的嵌入将输入标记和输出标记转换为维 $d_{model}$ 的向量. 我们还使用通常学习到的线性变换和 $softmax$ 函数将解码器输出转换为预测的下一个 token 概率. 在我们的模型中, 我们在两个嵌入层和 pre-softmax 线性变换之间共享相同的权重矩阵. 在嵌入层中, 我们将这些权重乘以 $\sqrt{d_{model}}$. 



### 3.5 位置编码

由于我们的模型不包含重复性和卷积, 因此为了使模型能够利用序列的顺序, 我们必须注入一些有关 token 在序列中的相对或绝对位置的信息. 为此, 我们在编码器和解码器底部的输入嵌入中添加 "位置编码". 位置编码的维数 $d_{model}$ 与嵌入的维数相同, 因此可以将二者相加. 位置编码有很多选择, 可以学习和固定. 

在这项工作中, 我们使用不同频率的正弦和余弦函数: 

$$\begin{aligned} PE_{(pos, 2i)} &= \sin(pos / 10000^{2i / d_{model}}) \\ PE_{(pos, 2i+1)} &= \cos(pos / 10000^{2i / d_{model}}) \end{aligned}$$ 

其中 $pos$ 是位置, $i$ 是尺寸. 即, 位置编码的每个维度对应于正弦曲线. 波长形成从 $2 \pi$ 到 $10000 \cdot 2 \pi$ 的几何级数. 我们选择此函数是因为我们假设它会允许模型轻松学习相对位置, 因为对于任何固定的偏移量 $k$, $PE_{pos + k}$ 可以表示为 $PE_{pos}$ 的线性函数. 

我们还尝试使用学习的位置嵌入进行实验, 发现这两个版本产生了几乎相同的结果. 我们选择正弦曲线版本是因为它可以使模型外推到比训练过程中遇到的序列长度更长的序列长度. 



练习实例: 

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def positional_encoding(embedding_size, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([
        pos/np.power(10000, 2*i/embedding_size)
        for pos in range(sentence_length)
        for i in range(embedding_size)
    ])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    ret = tf.convert_to_tensor(
        encoded_vec.reshape([sentence_length, embedding_size]),
        dtype=dtype
    )
    return ret


with tf.Session() as sess:
    positional_embedding = positional_encoding(5, 3)
    ret = sess.run(positional_embedding)
    print(ret)

```



以上代码输出结果为: 

```text
[[0.0000000e+00 1.0000000e+00 0.0000000e+00 1.0000000e+00 0.0000000e+00]
 [5.4030228e-01 2.5116222e-02 9.9999982e-01 1.5848931e-05 1.0000000e+00]
 [9.0929741e-01 9.9873835e-01 1.2619144e-03 1.0000000e+00 7.9621435e-07]]
```







### 4 Why Self-Attention

在本节中, 我们将自我注意层的各个方面与通常用于将一个可变长度的符号表示序列 $(x_{1}, \cdots , x_{n})$ 映射到另一个长度相等的序列 $(z_{1}, \cdots , z_{n})$ , 其中 $x_{i}, z_{i} \in \mathbb{R}^{d}$ 就像典型序列转换编码器或解码器中的隐藏层. 为了激发我们的自我注意力, 我们考虑了三个愿望. 

一种是每层的总计算复杂度. 另一个是可以并行化的计算量, 以所需的最少顺序操作数衡量. 

第三个是网络中远程依赖关系之间的路径长度. 在许多序列转导任务中, 学习远程依赖性是一项关键挑战. 影响学习这种依赖性的能力的一个关键因素是前向和后向信号必须在网络中穿越的路径长度. 输入和输出序列中位置的任意组合之间的这些路径越短, 学习远程依赖关系就越容易. 因此, 我们还比较了由不同层类型组成的网络中任意两个输入和输出位置之间的最大路径长度. 

如表 1 所示, 自我注意层使用恒定数量的顺序执行的操作连接所有位置, 而递归层则需要 $O(n)$ 个顺序操作. 就计算复杂度而言, 当序列长度 $n$ 小于表示维数 $d$ 时, 自注意力层比循环层要快, 这是机器翻译中最新模型所使用的句子表示最常见的情况, 例如 word-piece 和 byte-pair 表示形式. 为了提高涉及非常长序列的任务的计算性能, 可以将自我注意限制为仅考虑输入序列中以各个输出位置为中心的大小为 $r$ 的邻域. 这会将最大路径长度增加到 $O(n=r)$. 我们计划在以后的工作中进一步研究这种方法. 

内核宽度为 $k < n$ 的单个卷积层无法连接所有成对的输入和输出位置. 这样做需要在连续内核的情况下堆叠 $O(n=k)$ 个卷积层, 在膨胀卷积情况下则需要 $O(\log_{k}(n))$, 增加网络中任意两个位置之间最长路径的长度. 卷积层通常比循环层贵 $k$ 倍. 但是, 可分离卷积将复杂度大大降低到 $O(k \cdot n \cdot d + n \cdot d^{2})$. 然而, 即使 $k=n$, 可分离卷积的复杂度也等于我们模型中采用的自注意力层和逐点前馈层的组合. 

作为附带的好处, 自我关注可以产生更多可解释的模型. 我们从模型中检查注意力分布, 并在附录中介绍和讨论示例. 各个注意头不仅可以清楚地学会执行不同的任务, 而且很多似乎表现出与句子的句法和语义结构有关的行为. 

















***

### 练习实例



### Encoder-Decoder 实现

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class AttentionHelper(object):

    @classmethod
    def get_lower_triangle_mask(cls, to_mask_shape):
        """
        :param to_mask_shape: tuple or list. 指示所生成的 mask, 的目标形状.
        :return:
        """
        rank = len(to_mask_shape)
        diag_vals = tf.ones(to_mask_shape[-2:])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        n = rank - 2
        lower_tri_m = tf.tile(
            tf.reshape(tril, [1] * n + tril.get_shape().as_list()),
            [*to_mask_shape[:n], 1, 1]
        )
        return lower_tri_m

    @classmethod
    def get_upper_triangle_mask(cls, to_mask_shape):
        lower_tri_m = cls.get_lower_triangle_mask(to_mask_shape)
        rank = len(to_mask_shape)
        t = np.arange(rank)
        upper_tri_m = tf.transpose(lower_tri_m, [*t[:-2], t[-2], t[-1]])
        return upper_tri_m

    @classmethod
    def merge_mask1_and_mask2(cls, mask1, mask2):
        t_dtype = mask1.dtype
        mask1 = tf.cast(tf.not_equal(mask1, 0), tf.float32)
        mask2 = tf.cast(tf.not_equal(mask2, 0), tf.float32)
        mask_bool = tf.cast(tf.equal(mask1 + mask2, 2), dtype=tf.bool)
        mask = tf.cast(mask_bool, dtype=t_dtype)
        return mask

    @classmethod
    def apply_mask(cls, tensor, mask, value):
        paddings = tf.ones_like(mask) * value
        result = tf.where(tf.equal(mask, 0), paddings, tensor)
        return result

    @classmethod
    def expand_mask(cls, mask, dim):
        rank = tf.rank(mask)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1] * rank.eval() + [dim])
        return mask


class Attention(AttentionHelper):
    L_TRI_MASK = 'lower_triangle_mask'
    U_TRI_MASK = 'upper_triangle_mask'
    NO_MASK = 'none'

    def __init__(self, num_heads=1, triangle_mask=None, dropout=0.1):
        """
        :param num_heads: scalar.
        :param triangle_mask: str. choices:
        {`lower_triangle_mask`, `upper_triangle_mask`, `none`}.
        """
        self.num_heads = num_heads
        if triangle_mask not in {
            self.L_TRI_MASK,
            self.U_TRI_MASK,
            self.NO_MASK,
            None
        }:
            raise ValueError(
                f'triangle mask parameter must be one of: '
                f'`{self.L_TRI_MASK}`, `{self.U_TRI_MASK}`, `{self.NO_MASK}`. '
            )
        self._triangle_mask = triangle_mask if triangle_mask is not None else self.NO_MASK
        self._dropout = dropout

        self.embedding_size = None
        # sequence 张量的类型.
        self._t_type = None
        self._seq_shape = None
        self._seq_rank = None

    def __call__(self, q, k, v, q_mask=None, k_mask=None):
        self._t_type = q.dtype
        self._seq_shape = tf.shape(q).eval()
        self._seq_rank = len(self._seq_shape)
        self.embedding_size = self._seq_shape[-1]

        if q_mask is None:
            q_mask = tf.constant(
                value=1, shape=self._seq_shape[:-1],
                dtype=self._t_type
            )
        if k_mask is None:
            k_mask = tf.constant(
                value=1, shape=self._seq_shape[:-1],
                dtype=self._t_type
            )

        if self.embedding_size % self.num_heads != 0:
            raise ValueError(
                f'the embedding-size must be multiple of num-of-heads, '
                f'however: embeding size: '
                f'{self.embedding_size}, num-of-heads: {self.num_heads}'
            )

        result = self._multi_heads(q, k, v, q_mask, k_mask)
        return result

    def _multi_heads(self, q, k, v, q_mask=None, k_mask=None):
        q_mask = tf.cast(q_mask, dtype=q.dtype)
        k_mask = tf.cast(k_mask, dtype=q.dtype)

        emd_q = tf.shape(q)[-1]
        emd_v = tf.shape(v)[-1]

        dk = (emd_q // self.num_heads).eval()
        dv = (emd_v // self.num_heads).eval()

        q_mask = self.expand_mask(q_mask, dk)
        k_mask = self.expand_mask(k_mask, dv)

        head_list = []
        if self.num_heads > 1:
            for i in range(self.num_heads):
                qs, ks, vs = self._linear_projection(q, k, v, dk, dv)
                head_i = self._scaled_dot_product(qs, ks, vs, q_mask, k_mask)
                head_list.append(head_i)
            heads = tf.concat(head_list, axis=-1)
        elif self.num_heads == 1:
            heads = self._scaled_dot_product(q, k, v, q_mask, k_mask)
        else:
            raise ValueError(
                f'num-of-heads: '
                f'{self.num_heads} should not less than 1. '
            )
        result = tf.nn.dropout(heads, keep_prob=1.0 - self._dropout)
        return result

    @staticmethod
    def _linear_projection(q, k, v, dk, dv):
        q = tf.layers.dense(q, dk, use_bias=False)
        k = tf.layers.dense(k, dk, use_bias=False)
        v = tf.layers.dense(v, dv, use_bias=False)
        return q, k, v

    def _scaled_dot_product(self, q, k, v, q_mask, k_mask):
        """
        同时兼容 qs, ks, vs 形状为:
        (batch, sequence_len, embedding_size)
        或:
        (batch, num_heads, sequence_len, embedding_size)
        """
        dk = tf.cast(tf.shape(q)[-1], self._t_type)
        o1 = tf.matmul(q, k, transpose_b=True)
        o2 = o1 / (dk ** 0.5)

        o2_shape = tf.shape(o2).eval()
        co_seq_m = tf.matmul(q_mask, k_mask, transpose_b=True)

        if self._triangle_mask == self.L_TRI_MASK:
            lower_tri_m = self.get_lower_triangle_mask(o2_shape)
            mask = self.merge_mask1_and_mask2(co_seq_m, lower_tri_m)
        elif self._triangle_mask == self.U_TRI_MASK:
            upper_tri_m = self.get_upper_triangle_mask(o2_shape)
            mask = self.merge_mask1_and_mask2(co_seq_m, upper_tri_m)
        else:
            mask_bool = tf.not_equal(co_seq_m, 0)
            mask = tf.cast(mask_bool, dtype=self._t_type)

        o2 = self.apply_mask(o2, mask, -1e9)
        o3 = tf.nn.softmax(o2)
        output = tf.matmul(o3, v) * q_mask
        return output


class FFN(object):
    """FFN class (Position-wise Feed-Forward Networks)"""

    def __init__(self, ffn_dim=2048, model_dim=512, dropout=0.1):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout

    def __call__(self, inputs):
        return self.dense_relu_dense(inputs)
        # return self.conv_relu_conv(inputs)

    def dense_relu_dense(self, inputs):
        output = tf.layers.dense(inputs, self.ffn_dim, activation=tf.nn.relu)
        output = tf.layers.dense(output, self.model_dim)
        return tf.nn.dropout(output, 1.0 - self.dropout)

    def conv_relu_conv(self, inputs):
        output = tf.layers.conv1d(
            inputs, filters=self.ffn_dim, kernel_size=1, padding='SAME')
        output = tf.nn.relu(output)
        output = tf.layers.conv1d(
            output, filters=self.model_dim, kernel_size=1, padding='SAME')
        return tf.nn.dropout(output, 1.0 - self.dropout)


class Encoder(object):
    def __init__(self, num_layers=6, num_heads=8, ffn_dim=2048, model_dim=512):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.model_dim = model_dim

    def __call__(self, inputs):
        for i in range(self.num_layers):
            o1 = self._self_attention(q=inputs, k=inputs, v=inputs)
            o2 = self._add_and_norm(inputs, o1)
            o3 = self._ffn(o2)
            inputs = self._add_and_norm(o2, o3)
        return inputs

    def _self_attention(self, q, k, v):
        attention = Attention(
            num_heads=self.num_heads,
            triangle_mask=Attention.NO_MASK
        )
        return attention(q, k, v)

    def _ffn(self, inputs):
        ffn = FFN(ffn_dim=self.ffn_dim, model_dim=self.model_dim)
        return ffn(inputs)

    def _add_and_norm(self, x, sub_layer_x):
        output = tf.add(x, sub_layer_x)
        output = tf.contrib.layers.layer_norm(output)
        return output


class Decoder(object):
    def __init__(self, num_layers=6, num_heads=8, ffn_dim=2048, model_dim=512):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.model_dim = model_dim

    def __call__(self, decoder_inputs, encoder_outputs):
        for i in range(self.num_layers):
            o1 = self._masked_self_attention(
                q=decoder_inputs,
                k=decoder_inputs,
                v=decoder_inputs
            )
            o2 = self._add_and_norm(decoder_inputs, o1)

            o3 = self._encoder_decoder_attention(
                q=o2,
                k=encoder_outputs,
                v=encoder_outputs
            )
            o4 = self._add_and_norm(o2, o3)

            o5 = self._ffn(o2)
            decoder_inputs = self._add_and_norm(o4, o5)
        return decoder_inputs

    def _masked_self_attention(self, q, k, v):
        attention = Attention(
            num_heads=self.num_heads,
            triangle_mask=Attention.L_TRI_MASK
        )
        return attention(q, k, v)

    def _encoder_decoder_attention(self, q, k, v):
        attention = Attention(
            num_heads=self.num_heads,
            triangle_mask=Attention.NO_MASK
        )
        return attention(q, k, v)

    def _ffn(self, inputs):
        ffn = FFN(ffn_dim=self.ffn_dim, model_dim=self.model_dim)
        return ffn(inputs)

    def _add_and_norm(self, x, sub_layer_x):
        output = tf.add(x, sub_layer_x)
        output = tf.contrib.layers.layer_norm(output)
        return output


def demo2():
    inputs = tf.constant(
        value=1,
        shape=(1, 10, 512),
        dtype=tf.float32
    )

    with tf.Session() as sess:
        encoder = Encoder()
        outputs = encoder(inputs)

        sess.run(tf.global_variables_initializer())

        ret = sess.run(outputs)
        print(ret)
        print(ret.shape)

    return


def demo1():
    inputs = tf.constant(
        value=1,
        shape=(1, 10, 512),
        dtype=tf.float32
    )

    with tf.Session() as sess:
        attention = Attention()
        outputs = attention(q=inputs, k=inputs, v=inputs)

        sess.run(tf.global_variables_initializer())

        ret = sess.run(outputs)
        print(ret)
        print(ret.shape)
    return


def demo3():
    inputs = tf.constant(
        value=1,
        shape=(1, 10, 512),
        dtype=tf.float32
    )

    with tf.Session() as sess:
        encoder = Encoder()
        decoder = Decoder()
        encoder_outputs = encoder(inputs)
        decoder_outputs = decoder(
            decoder_inputs=inputs, 
            encoder_outputs=encoder_outputs
        )
        sess.run(tf.global_variables_initializer())

        ret = sess.run(decoder_outputs)
        print(ret)
        print(ret.shape)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()

```



以上代码输出结果为: 

```text
# demo1()
[[[1.1111113 1.1111113 1.1111113 ... 1.1111113 1.1111113 1.1111113]
  [1.1111113 1.1111113 1.1111113 ... 1.1111113 1.1111113 1.1111113]
  [1.1111113 1.1111113 1.1111113 ... 1.1111113 1.1111113 1.1111113]
  ...
  [1.1111113 1.1111113 1.1111113 ... 1.1111113 1.1111113 1.1111113]
  [1.1111113 1.1111113 1.1111113 ... 1.1111113 1.1111113 1.1111113]
  [1.1111113 1.1111113 0.        ... 1.1111113 1.1111113 1.1111113]]]
(1, 10, 512)

# demo2()
[[[-1.7595223   1.2685192  -0.68852013 ...  1.4378083  -1.096309
   -0.81514996]
  [-2.2212849   1.1076035  -0.4784988  ...  1.4491912  -0.84780914
   -0.96534115]
  [-1.8761137   0.86478394 -0.7461215  ...  1.7201781  -0.2718768
   -0.9030095 ]
  ...
  [-2.0132048   1.2104907  -0.70444274 ...  1.2241931  -0.9466536
   -1.2563595 ]
  [-2.649755    0.49965477 -0.7386076  ... -0.02871626 -1.6344287
   -1.259026  ]
  [-2.160051    1.1489758  -0.35147122 ...  0.8749433  -0.92735714
   -1.4229223 ]]]
(1, 10, 512)

# demo3()
[[[-0.700255    0.48235515  0.80225724 ...  0.24289286  0.3277873
   -0.78373283]
  [-0.8879549   0.05303033  1.3356409  ... -0.15840939  0.44428286
   -0.7694273 ]
  [-0.85245645 -0.42307475  0.96337676 ... -0.18543954  0.5074707
   -0.9304163 ]
  ...
  [-1.076197    0.23550019  1.1743187  ...  0.2620472   0.8970079
   -1.1169561 ]
  [-1.1077378  -0.13534367  0.86359996 ... -0.5329288   0.7159704
   -0.68765146]
  [-0.821921    0.2223114   1.3377131  ... -0.05079698  0.7205519
   -0.9025247 ]]]
(1, 10, 512)
```







