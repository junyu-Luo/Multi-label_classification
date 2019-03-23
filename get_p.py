import os

import keras
import tensorflow as tf
# import ujson as json
import json
import numpy as np
import pandas as pd

# from NERandNRE.settings import BASE_DIR
from prepro import convert_to_features
from layers import dense, native_rnn, native_birnn, cudnn_rnn, cudnn_birnn, bidaf, total_params

tf.reset_default_graph()  # 迷之重置  否则apache 报错
pretrained_emb = True
trainable_emb = False
if not pretrained_emb:
    trainable_emb = True

BASE_DIR = './'
flags = tf.flags
flags.DEFINE_integer("test_x_limit", 200, "Limit length for paragraph")
flags.DEFINE_integer("test_q_limit", 22, "Limit length for question")
flags.DEFINE_integer("hidden", 100, "Hidden size")
flags.DEFINE_integer("num_layers", 1, "Hidden size")
flags.DEFINE_boolean("birnn", False, "Whether to use birnn")
flags.DEFINE_boolean("cudnn", False, "Whether to use cudnn rnn cell")
flags.DEFINE_boolean("trainable_emb", trainable_emb, "Whether to train pretrained embedding")
flags.DEFINE_boolean("bidaf", True, "Whether to use bidaf")
flags.DEFINE_float("gamma", 2., "gamma of focal loss")


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

            self.pre_q = q
            self.pre_q_mask = self.q_mask

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


def demo(context, config=flags.FLAGS):
    keras.backend.clear_session()  # 清除原本数据
    emb_mat_file = ("prepro_/pretrained/joint_joint/02/emb_mat.json")
    emb_mat_file = BASE_DIR + '/' + emb_mat_file
    token2idx_file = ("prepro_/pretrained/joint_joint/02/token2idx.json")
    token2idx_file = BASE_DIR + '/' + token2idx_file
    save_dir = ("results/joint_jointdeploy/model/02/")
    save_dir = BASE_DIR + '/' + save_dir

    with open(emb_mat_file, 'r') as fh:
        emb_mat = np.array(json.load(fh), dtype=np.float32)
    with open(token2idx_file, 'r') as fh:
        token2idx = json.load(fh)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    sess = tf.Session(config=sess_config)
    model = Model(config, None, emb_mat, trainable=False, demo=True)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    # saver.restore(sess, os.path.join(save_dir, 'model_{}.ckpt'.format(43830)))
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    def qqqaa_v2(context):

        c, q, context_num, batch_size = convert_to_features(config, context, token2idx)
        fd = {'context:0': c,
              'query:0': q,
              'batch_size:0': batch_size}
        q_mask, q, pred = sess.run([model.pre_q_mask, model.pre_q, model.pred], feed_dict=fd)
        from six.moves import cPickle as pickle
        with open('q.pkl', 'wb') as f:
            pickle.dump(q, f)
        with open('q_mask.pkl', 'wb') as f:
            pickle.dump(q_mask, f)
        pred = np.sum(np.array(pred).reshape([context_num, 10]), 0)
        pred = ''.join(pred.astype(bool).astype(int).astype(str))
        sick_list = [
            '慢支炎',
            '肺气肿',
            '肺动脉高压',
            '肺部感染',
            '纤维灶',
            '支气管',
            '间质肺',
            '主动脉硬化',
            '空洞 / 空腔',
            '肺结节影',
        ]
        pred = list(pred)
        # num = 0
        result = {}
        for i in range(len(pred)):
            if pred[i] == '1':
                result[sick_list[i]] = '有'
            else:
                result[sick_list[i]] = '无'
            # num+=1
        return result

    return qqqaa_v2(context)

    # def do_file(filename):
    #     root_path = '.'
    #     # with open(os.path.join(root_path,'CT.xls'),'r',encoding='gbk')as wf:
    #     data = pd.read_excel(os.path.join(root_path,filename))
    #     REPORTDESCRIBE_columns = ['INPATIENT_NO','REPORTDESCRIBE','（慢支炎征象）肺纹理增多/增粗/紊乱',
    #                                                    '（肺气肿征象）透亮度增加/膈肌低平/肺大泡',
    #                                                    '（肺动脉高压征象）右下肺动脉增宽/肺动脉段突出/右室增大',
    #                                                    '（肺部感染征象）单发或多发斑片状阴影',
    #                                                    '（陈旧性肺结核征象）/纤维条索影',
    #                                                    '（支气管扩张征象）卷发状阴影/囊状透光区/环形阴影/轨道征',
    #                                                    '（间质肺征象）磨玻璃/网格状/蜂窝状阴影',
    #                                                    '主动脉硬化，主动脉钙化',
    #                                                    '空洞/空腔',
    #                                                    '肺结节影']
    #     REPORTDESCRIBE_outfile = pd.DataFrame(columns=REPORTDESCRIBE_columns)
    #     REPORTDIAGNOSE_columns = ['INPATIENT_NO','REPORTDIAGNOSE','（慢支炎征象）肺纹理增多/增粗/紊乱',
    #                                                    '（肺气肿征象）透亮度增加/膈肌低平/肺大泡',
    #                                                    '（肺动脉高压征象）右下肺动脉增宽/肺动脉段突出/右室增大',
    #                                                    '（肺部感染征象）单发或多发斑片状阴影',
    #                                                    '（陈旧性肺结核征象）/纤维条索影',
    #                                                    '（支气管扩张征象）卷发状阴影/囊状透光区/环形阴影/轨道征',
    #                                                    '（间质肺征象）磨玻璃/网格状/蜂窝状阴影',
    #                                                    '主动脉硬化，主动脉钙化',
    #                                                    '空洞/空腔',
    #                                                    '肺结节影']
    #     REPORTDIAGNOSE_outfile = pd.DataFrame(columns=REPORTDIAGNOSE_columns)
    #     for i in range(len(data['REPORTDESCRIBE'])):
    #         context = str(data['REPORTDESCRIBE'][i])
    #         c, q, context_num, batch_size = convert_to_features(config, context, token2idx)
    #         fd = {'context:0': c,
    #               'query:0': q,
    #               'batch_size:0': batch_size}
    #         pred = sess.run([model.pred], feed_dict=fd)
    #         pred = np.sum(np.array(pred).reshape([context_num, 10]), 0)
    #         pred = ''.join(pred.astype(bool).astype(int).astype(str))
    #         pred = list(pred)
    #         REPORTDESCRIBE_column = []
    #         # REPORTDESCRIBE_column.append(data['INPATIENT_NO'][i])
    #         REPORTDESCRIBE_column.append(i)
    #         REPORTDESCRIBE_column.append(data['REPORTDESCRIBE'][i])
    #         REPORTDESCRIBE_column.extend(pred)
    #         REPORTDESCRIBE_outfile = REPORTDESCRIBE_outfile.append(pd.DataFrame([REPORTDESCRIBE_column],columns=REPORTDESCRIBE_columns),ignore_index=True)
    #
    #         context2 = str(data['REPORTDIAGNOSE'][i])
    #         c, q, context_num, batch_size = convert_to_features(config, context2, token2idx)
    #         fd = {'context:0': c,
    #               'query:0': q,
    #               'batch_size:0': batch_size}
    #         pred2 = sess.run([model.pred], feed_dict=fd)
    #         pred2 = np.sum(np.array(pred2).reshape([context_num, 10]), 0)
    #         pred2 = ''.join(pred2.astype(bool).astype(int).astype(str))
    #         pred2 = list(pred2)
    #         REPORTDIAGNOSE_column = []
    #         # REPORTDIAGNOSE_column.append(data['INPATIENT_NO'][i])
    #         REPORTDIAGNOSE_column.append(i)
    #         REPORTDIAGNOSE_column.append(data['REPORTDIAGNOSE'][i])
    #         REPORTDIAGNOSE_column.extend(pred2)
    #         REPORTDIAGNOSE_outfile = REPORTDIAGNOSE_outfile.append(pd.DataFrame([REPORTDIAGNOSE_column],columns=REPORTDIAGNOSE_columns),ignore_index=True)
    #
    #     REPORTDESCRIBE_outfile.to_excel("outputdata/outfile.xlsx")

    # return do_file(filename)


if __name__ == '__main__':
    yn_zhenduan = """ 1、双肺散在斑片状密度增高影，考虑为感染性病变。
    2、慢性支气管炎表现。
    3、左心室增大、肺动脉增宽。
    4、纵隔淋巴结显示。
    5、双侧胸膜腔少量积液，双侧胸膜增厚、粘连。
    6、扫描层面显示胆囊炎、胆结石。"""
    yn_zhenduan = '慢支炎。'
    a = demo(yn_zhenduan)
    print(a)

# config = flags.FLAGS
# filename = 'yn_data.xls'
# demo(config, filename)
