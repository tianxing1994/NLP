#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.attention.encoder_block import EncoderBlock
from tf_tools.attention.layer_normalization import LayerNormalization
from tf_tools.attention.positional_embedding import PositionalEmbedding
from tf_tools.common.embedding import EmbeddingLookup
from tf_tools.common.dense import Dense
from tf_tools.debug_tools.common import session_run


class Bert(object):
    def __init__(self, vocab_size, embedding_size, dropout, max_len, hidden_size,
                 n_hidden_layers, share_encoder_block,
                 heads=2, name='bert', reuse=tf.AUTO_REUSE):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.share_encoder_block = share_encoder_block
        self.heads = heads

        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

        self.token_embedding_lookup = EmbeddingLookup(
            input_dim=vocab_size,
            output_dim=embedding_size,
            name='embedding_token'
        )

        self.segment_embedding_lookup = EmbeddingLookup(
            input_dim=2,
            output_dim=embedding_size,
            name='embedding_segment'
        )

        self.positional_embedding = PositionalEmbedding(
            input_dim=self.max_len,
            output_dim=self.embedding_size,
            embedding_type=PositionalEmbedding.t_random,
            merge_mode=PositionalEmbedding.m_add,
            name='positional_embedding'
        )

        self.layer_normalization = LayerNormalization(name='layer_normalization')

        if self.embedding_size != self.hidden_size:
            self.dense = Dense(
                units=hidden_size,
                use_bias=True,
                name='dense'
            )

    def __call__(self, tokens, segment, mask=None):
        with self.variable_scope:
            with self.name_scope:
                x = tokens
                s = segment
                if mask is None:
                    mask = tf.cast(tf.greater(x, 0), dtype=tf.float32, name='sequence_mask')
                x = self.token_embedding_lookup(x)
                s = self.segment_embedding_lookup(s)
                x = tf.add(x, s, name='add')
                x = self.positional_embedding(x)
                x = self.layer_normalization(x)
                if self.dropout > 0:
                    x = tf.nn.dropout(x, keep_prob=1 - self.dropout)
                if self.embedding_size != self.hidden_size:
                    x = self.dense(x)

                encoder_block = None
                for i in range(self.n_hidden_layers):
                    if self.share_encoder_block is False or encoder_block is None:
                        name = 'encoder_block' \
                            if self.share_encoder_block \
                            else f'encoder_block_{i+1}'
                        encoder_block = EncoderBlock(
                            heads=self.heads,
                            dropout=self.dropout,
                            ffn_dim=self.hidden_size,
                            model_dim=self.hidden_size,
                            name=name
                        )
                    x = encoder_block(inputs=x, mask=mask)
        return x


def demo1():
    """
    [PAD]: 0
    [CLS]: 1
    [SEP]: 2
    :return:
    """
    x = np.array(
        [[1, 8, 26, 15, 28, 21, 4, 10, 4, 9, 4, 0, 0, 0, 0, 0],
         [1, 87, 6, 53, 81, 45, 4, 10, 4, 6, 0, 0, 0, 0, 0, 0],
         [1, 2, 61, 22, 78, 45, 4, 10, 4, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int
    )
    y = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int
    )

    # shape=(batch_size, seq_len). seq_len ???????????? max_len ??????,
    # ???????????????????????????????????? Bert ?????????????????????, ????????? seq_len ??????.
    tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name='input_tokens')
    segment = tf.placeholder(dtype=tf.int32, shape=(None, None), name='input_segment')

    bert = Bert(
        vocab_size=100,
        embedding_size=64,
        max_len=64,
        dropout=0.1,
        hidden_size=128,
        n_hidden_layers=6,
        share_encoder_block=True,
        name='bert'
    )

    outputs = bert(tokens, segment)
    ret = session_run(outputs, feed_dict={tokens: x, segment: y})
    # print(ret)
    # print(ret.shape)

    x = ret[:, 0]
    print(x)
    print(x.shape)
    print(type(x))
    return


def demo2():
    """
    [PAD]: 0
    [CLS]: 1
    [SEP]: 2
    :return:
    """
    x = np.array(
        [1, 8, 26, 15, 28, 21, 4, 10, 4, 9, 4, 0, 0, 0, 0, 0],
        dtype=np.int
    )
    y = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=np.int
    )

    tokens = tf.placeholder(dtype=tf.int32, shape=(None,), name='input_tokens')
    segment = tf.placeholder(dtype=tf.int32, shape=(None,), name='input_segment')

    bert = Bert(
        vocab_size=100,
        embedding_size=64,
        max_len=64,
        dropout=0.1,
        hidden_size=128,
        n_hidden_layers=6,
        share_encoder_block=True,
        name='bert'
    )

    outputs = bert(tokens, segment)
    ret = session_run(outputs, feed_dict={tokens: x, segment: y})
    # print(ret)
    # print(ret.shape)

    x = ret[:, 0]
    print(x)
    print(len(x))
    print(type(x))
    return


if __name__ == '__main__':
    demo1()
    # demo2()
