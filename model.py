import tensorflow as tf
from layers import dense, native_rnn, native_birnn, cudnn_rnn, cudnn_birnn, bidaf, total_params


class Model():
    def __init__(self, config, iterator, emb_mat, trainable=True, opt=True, demo=False):
        self.config = config
        self.emb_mat = emb_mat
        self.demo = demo

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")

        if self.demo:
            self.c = tf.placeholder(tf.int32, [None, config.test_x_limit], "context")
            self.q = tf.placeholder(tf.int32, [None, config.test_q_limit], "query")
            self.y = tf.placeholder(tf.float32, [None], "y")
            self.batch_size = tf.placeholder(tf.int32, None, "batch_size")
        else:
            self.c, self.q, self.y = iterator.get_next()

        with tf.variable_scope("opt"):
            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

            if opt:
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [-1, self.c_maxlen])
                self.q = tf.slice(self.q, [0, 0], [-1, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [-1, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [-1, self.q_maxlen])
            else:
                self.c_maxlen, self.q_maxlen = config.x_limit, config.q_limit

        self._build_model()
        if not config.cudnn:
            total_params()

        if trainable:
            if config.l2_norm:
                regularizer = tf.contrib.layers.l2_regularizer(config.l2_norm)
                variables = tf.trainable_variables()
                variables = [v for v in variables if "bias" not in v.name]  # don't regularize bias
                self.l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                self.loss += self.l2_loss
            # self.loss -= self.l2_loss

            # optimizer
            self.lr = tf.placeholder_with_default(0.001, (), name="lr")
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            if config.grad_clip_flag:
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                    zip(capped_grads, variables), global_step=self.global_step)
            else:
                self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

            # ema
            if config.decay:
                self.ema = tf.train.ExponentialMovingAverage(config.decay)
                ema_op = self.ema.apply(tf.trainable_variables())
                with tf.control_dependencies([self.train_op]):
                    self.train_op = tf.group(ema_op)

    def _build_model(self):
        config = self.config
        N, PL, QL, d, nl = config.batch_size if not self.demo else self.batch_size, self.c_maxlen, self.q_maxlen, config.hidden, config.num_layers
        if config.birnn:
            rnn = cudnn_birnn if config.cudnn else native_birnn
        else:
            rnn = cudnn_rnn if config.cudnn else native_rnn

        with tf.variable_scope("embed"):
            self.emb_mat = tf.get_variable("emb_mat", initializer=tf.constant(
                self.emb_mat, dtype=tf.float32), trainable=config.trainable_emb)
            c_emb = tf.nn.embedding_lookup(self.emb_mat, self.c)
            q_emb = tf.nn.embedding_lookup(self.emb_mat, self.q)

        with tf.variable_scope("encoder"):
            c_emb = tf.nn.dropout(c_emb, 1.0 - self.dropout, noise_shape=[N, 1, c_emb.shape[2]])
            q_emb = tf.nn.dropout(q_emb, 1.0 - self.dropout, noise_shape=[N, 1, q_emb.shape[2]])

            # !!!use same rnn?!!!
            # c = rnn(c_emb, nl, d, self.c_len, name='encoding_rnn_c', reuse=None)
            # q = rnn(q_emb, nl, d, self.q_len, name='encoding_rnn_q', reuse=None)
            c = rnn(c_emb, nl, d, self.c_len, name='encoding_rnn', reuse=None)
            q = rnn(q_emb, nl, d, self.q_len, name='encoding_rnn', reuse=True)

        with tf.variable_scope("attention"):
            c_att = bidaf(c, q, q, self.c_mask, self.q_mask, q2c=config.bidaf)

        with tf.variable_scope("prediction"):
            # c_att = tf.nn.dropout(c_att, 1.0 - self.dropout, noise_shape=[N, 1, c_att.shape[2]])
            # c_att = rnn(c_att, nl, d, self.c_len, name='output_rnn', reuse=None)
            # if config.cudnn:
            # 	c_att = c_att * tf.cast(tf.expand_dims(self.c_mask, 2), tf.float32)

            c_out = tf.reduce_max(c_att, 1)

            pred = dense(c_out, 1, 'sigmoid')
            pred = tf.squeeze(pred, squeeze_dims=[1])

            self.loss = -tf.reduce_mean(
                self.y * (1. - pred) ** config.gamma * tf.log(pred) + (1. - self.y) * (pred) ** config.gamma * tf.log(
                    1. - pred))  # focal loss

            pred_ = tf.greater(pred, 0.5)
            label_ = tf.cast(self.y, tf.bool)
            self.pred = pred_
            self.label = label_
            self.acc = tf.reduce_mean(tf.cast(tf.equal(pred_, label_), tf.float32))

            self.tp = tf.reduce_sum(tf.cast(tf.logical_and(pred_, label_), tf.float32))
            self.fp = tf.reduce_sum(tf.cast(tf.logical_and(pred_, tf.logical_not(label_)), tf.float32))
            self.fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(pred_), label_), tf.float32))
        # self.precision = tp / (tp + fp + 1e-6)
        # self.recall = tp / (tp + fn + 1e-6)
