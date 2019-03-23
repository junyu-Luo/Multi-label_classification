import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import initializers


def get_keras_activation(name):
    return activations.get(name)


def _wrap_init(init_fn):
    def wrapped(shape, dtype=None, partition_info=None):
        if partition_info is not None:
            raise ValueError()
        return init_fn(shape, dtype)

    return wrapped


def get_keras_initialization(name):
    if name is None:
        return None
    return _wrap_init(initializers.get(name))


def dense(x, units, activation=None, bias=True, name="dense", reuse=None):
    fn = get_keras_activation(activation) if activation else None
    return tf.layers.dense(x, units, fn, bias, name=name, reuse=reuse)


def native_rnn(x, num_layers, num_units, seq_len, kind="GRU", concat_layers=False, name="rnn", reuse=None):
    assert kind in ["GRU", "LSTM"]
    res = [x]
    with tf.variable_scope(name, reuse=reuse):
        for i in range(num_layers):
            with tf.variable_scope("layer_{}".format(i)):
                if kind == "GRU":
                    cell = tf.contrib.rnn.GRUCell(num_units)
                else:
                    cell = tf.contrib.rnn.LSTMCell(num_units)
                outputs, _ = tf.nn.dynamic_rnn(cell, res[-1], seq_len, dtype=tf.float32)
                res.append(outputs)
        if concat_layers:
            return tf.concat(res[1:], axis=-1)
        else:
            return res[-1]


def cudnn_rnn(x, num_layers, num_units, seq_len, kind="GRU", concat_layers=False, name="cudnn_rnn", reuse=None):
    assert kind in ["GRU", "LSTM"]
    outputs = [tf.transpose(x, [1, 0, 2])]
    with tf.variable_scope(name, reuse=reuse):
        for i in range(num_layers):
            with tf.variable_scope("layer_{}".format(i)):
                if kind == "GRU":
                    rnn = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, dtype=tf.float32)
                else:
                    rnn = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units, dtype=tf.float32)

                out, _ = rnn(outputs[-1])
                outputs.append(out)
        if concat_layers:
            res = tf.concat(outputs[1:], axis=-1)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


def native_birnn(x, num_layers, num_units, seq_len, kind="GRU", concat_layers=False, name="birnn", reuse=None):
    assert kind in ["GRU", "LSTM"]
    res = [x]
    with tf.variable_scope(name, reuse=reuse):
        for i in range(num_layers):
            with tf.variable_scope("layer_{}".format(i)):
                if kind == "GRU":
                    fw_cell = tf.contrib.rnn.GRUCell(num_units)
                    bw_cell = tf.contrib.rnn.GRUCell(num_units)
                else:
                    fw_cell = tf.contrib.rnn.LSTMCell(num_units)
                    bw_cell = tf.contrib.rnn.LSTMCell(num_units)

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, res[-1], seq_len, dtype=tf.float32)
                res.append(tf.concat(outputs, axis=-1))
        if concat_layers:
            return tf.concat(res[1:], axis=-1)
        else:
            return res[-1]


def cudnn_birnn(x, num_layers, num_units, seq_len, kind="GRU", concat_layers=False, name="cudnn_birnn", reuse=None):
    assert kind in ["GRU", "LSTM"]
    outputs = [tf.transpose(x, [1, 0, 2])]
    with tf.variable_scope(name, reuse=reuse):
        for i in range(num_layers):
            with tf.variable_scope("layer_{}".format(i)):
                if kind == "GRU":
                    fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, dtype=tf.float32)
                    bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, dtype=tf.float32)
                else:
                    fw = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units, dtype=tf.float32)
                    bw = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units, dtype=tf.float32)

                with tf.variable_scope("fw_{}".format(i)):
                    out_fw, _ = fw(outputs[-1])
                with tf.variable_scope("bw_{}".format(i)):
                    inputs_bw = tf.reverse_sequence(outputs[-1], seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    out_bw, _ = bw(inputs_bw)
                    out_bw = tf.reverse_sequence(out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                outputs.append(tf.concat([out_fw, out_bw], axis=-1))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=-1)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


def _get_scores(x, keys, bias=True, init="glorot_uniform"):
    init = get_keras_initialization(init)

    key_w = tf.get_variable("key_w", shape=keys.shape.as_list()[-1], initializer=init, dtype=tf.float32)
    key_logits = tf.tensordot(keys, key_w, axes=[[2], [0]])  # (batch, key_len)

    x_w = tf.get_variable("input_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)
    x_logits = tf.tensordot(x, x_w, axes=[[2], [0]])  # (batch, x_len)

    dot_w = tf.get_variable("dot_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)

    # Compute x * dot_weights first, the batch mult with x
    x_dots = x * tf.expand_dims(tf.expand_dims(dot_w, 0), 0)
    dot_logits = tf.matmul(x_dots, keys, transpose_b=True)

    res = dot_logits + tf.expand_dims(key_logits, 1) + tf.expand_dims(x_logits, 2)
    if bias:
        res += tf.get_variable("bias", shape=(), dtype=tf.float32)
    return res


def softmax_with_mask(logits, mask, axis=-1):
    mask = tf.cast(mask, logits.dtype)
    e = tf.exp(logits) * mask
    s = tf.reduce_sum(e, axis=axis, keepdims=True)
    s = tf.clip_by_value(s, np.finfo(np.float32).eps, np.finfo(np.float32).max)
    return e / s


def reduce_max_with_mask(x, mask, axis=None):
    x = x * tf.cast(mask, x.dtype)
    return tf.reduce_max(x, axis=axis)


def bidaf(x, keys, memories, x_mask, mem_mask, q2c=True, query_dots=True, name="bidaf", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        dist_matrix = _get_scores(x, keys)  # (batch, x_words, keys_words)
        joint_mask = tf.logical_and(tf.expand_dims(x_mask, 2), tf.expand_dims(mem_mask, 1))
        # dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))
        # query_probs = tf.nn.softmax(dist_matrix)  # probability of each mem_word per x_word
        query_probs = softmax_with_mask(dist_matrix, joint_mask)

        # Batch matrix multiplication to get the attended vectors
        select_query = tf.matmul(query_probs, memories)  # (batch, x_words, q_dim)

        if not q2c:
            if query_dots:
                return tf.concat([x, select_query, x * select_query], axis=2)
            else:
                return tf.concat([x, select_query], axis=2)

        # select query-to-context
        context_dist = reduce_max_with_mask(dist_matrix, joint_mask, axis=2)  # (batch, x_words)
        context_probs = softmax_with_mask(context_dist, x_mask)  # (batch, x_words)
        select_context = tf.einsum("ai,aik->ak", context_probs, x)  # (batch, x_dim)
        select_context = tf.expand_dims(select_context, 1)

        # inconformable with paper
        if query_dots:
            return tf.concat([x, select_query, x * select_query, x * select_context], axis=2)
        else:
            return tf.concat([x, select_query, x * select_context], axis=2)


def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
