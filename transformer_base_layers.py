"""
Module for defining tensorflow models used for practice wrapping my mind around attention.

Word embeddings will be calculated using FastText embedding. I'm too lazy to figure out how word-piece works,
and hopefully this will save a few million training parameters.

Code heavily inspired by conventions used in source code for BERT

"""



import tensorflow as tf
import numpy as np
from gensim.models.fasttext import FastText

@tf.function
def gelu(x):
    """ Activation function used in BERT, a third-order approximation of gaussian error activation """
    cdf = 0.5 * ( 1 + tf.tanh(
        tf.sqrt(2) / np.pi * (x + 0.44715*tf.pow(x, 3))
    ))
    return x * cdf


@tf.function
def tensor_to_matrix(input_tensor):
    """
    Convert tensor of shape (n_samples, seq_len, emb_dim) to shape (n_samples*seq_len, emb_dim)
    :param input_tensor: tf.tensor
    :return: tf.tensor, will have 2 dimensions
    """
    emb_dim = tf.shape(input_tensor)[-1]
    out_tensor = tf.reshape(input_tensor, [-1, emb_dim])
    return out_tensor

@tf.function
def expand_for_attention(input_tensor, batch_size, n_attention_heads, seq_len, emb_dim):
    """
    Expand tensor before passing into attention computation
    :param input_tensor:
    :param batch_size:
    :param n_attention_heads:
    :param seq_len:
    :param emb_dim:
    :return:
    """


@tf.function
def attention_layer(from_tensor, to_tensor, n_attention_heads=1, size_per_head=512, mask=None):
    """
    Return scaled dot-product attention.
    :param from_tensor: tf.tensor of shape (batch_size, from_seq_len, emb_dim), from tensor for embeddings
    :param to_tensor: tf.tensor of shape (batch_size, from_seq_len, emb_dim), to tensor for embeddings
    :param n_attention_heads: int, number of attention heads to use for computation
    :param size_per_head: int, size to use for each attention head
    :param mask: tf.tensor of shape (from_seq_len, to_seq_len), 0-1 mask of what values should be attended to
    :return: A, tf.tensor of shape (batch_size, from_seq_len, to_seq_len), attention values for each key-query pair
    """

    # xx_tensor_2d [batch_size * xx_seq_len, emb_dim]
    from_tensor_2d = tensor_to_matrix(from_tensor)
    to_tensor_2d = tensor_to_matrix(to_tensor)

    query_layer = tf.keras.layers.Dense(n_attention_heads * size_per_head, activation=None)
    key_layer = tf.keras.layers.Dense(n_attention_heads * size_per_head, activation=None)
    value_layer = tf.keras.layers.Dense(n_attention_heads * size_per_head, activation=None)

    # query [batch_size * from_seq_len, n_attention_heads * size_per_head]
    query = query_layer(to_tensor_2d)
    query
    #key [batch_size *

    key_layer = tf.


    # dot_production [n_samples, from_seq_len, to_seq_len], scaled by dimension
    attention_scores = tf.matmul(Q, K) / tf.sqrt(emb_dim)

    # if mask, then set all attention values where mask is False to -1000 before softmax
    if mask:
        mask = tf.expand_dims(mask, axis=[0])
        mask = 1 - mask
        mask = tf.multiply(mask, -10000)
        mask = tf.cast(mask, tf.float32)
        attention_scores += mask

    # softmax [n_sample, seq_len, seq_len]
    attention_probas = tf.nn.softmax(attention_scores, axis=-1)

    # context [n_samples, seq_len, emb_dim]
    context = tf.matmul(attention_probas, V)

    return context


