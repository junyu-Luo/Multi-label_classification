import os

import keras
import tensorflow as tf
import json
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
# from Common.settings import BASE_DIR
from prepro import convert_to_features
from layers import dense, native_rnn, native_birnn, cudnn_rnn, cudnn_birnn, bidaf, total_params
import time
from tqdm import trange


def time_cal(func):
    def wrapper(*args, **kwargs):
        start = time.clock()
        result = func(*args, **kwargs)
        end = time.clock()
        print("%s running time:%s s" % (func.__name__, end - start))
        return result

    return wrapper


tf.reset_default_graph()  # 迷之重置  否则apache 报错
pretrained_emb = True
trainable_emb = False
if not pretrained_emb:
    trainable_emb = True


flags = tf.flags
flags.DEFINE_integer("test_x_limit", 200, "Limit length for paragraph")
flags.DEFINE_integer("test_q_limit", 172, "Limit length for question")
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
            self.c = tf.placeholder(tf.int32, [1, config.test_x_limit], "context")
            self.q = tf.placeholder(tf.int32, [10, config.test_q_limit], "query")
            self.y = tf.placeholder(tf.float32, [None], "y")
            self.batch_size = tf.placeholder(tf.int32, None, "batch_size")
            self.pre_q = tf.placeholder(tf.float32, [10, config.test_q_limit, config.hidden], "pre_query")
            self.pre_q_mask = tf.placeholder(tf.bool, [10, config.test_q_limit], "pre_query_mask")
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

            # tile c to match q's size
            c = tf.tile(c, [10, 1, 1])

            if self.demo:
                q = self.pre_q
                self.q_mask = self.pre_q_mask
            else:
                q = rnn(q_emb, nl, d, self.q_len, name='encoding_rnn', reuse=True)

        with tf.variable_scope("attention"):
            c_att = bidaf(c, q, q, self.c_mask, self.q_mask, q2c=config.bidaf)

        with tf.variable_scope("prediction"):
            c_out = tf.reduce_max(c_att, 1)
            pred = dense(c_out, 1, 'sigmoid')
            # pred = tf.Print(pred,[pred],'1111') #print

            pred = tf.squeeze(pred, squeeze_dims=[1])
            # pred = tf.Print(pred, [pred], 'print pred：')  # print

            self.loss = -tf.reduce_mean(
                self.y * (1. - pred) ** config.gamma * tf.log(pred) + (1. - self.y) * (pred) ** config.gamma * tf.log(
                    1. - pred))  # focal loss

            pred_ = tf.greater(pred, 0.5)
            label_ = tf.cast(self.y, tf.bool)
            self.pred_probability = pred
            self.pred = pred_
            self.label = label_
            self.acc = tf.reduce_mean(tf.cast(tf.equal(pred_, label_), tf.float32))
            self.tp = tf.reduce_sum(tf.cast(tf.logical_and(pred_, label_), tf.float32))
            self.fp = tf.reduce_sum(tf.cast(tf.logical_and(pred_, tf.logical_not(label_)), tf.float32))
            self.fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(pred_), label_), tf.float32))
        # self.precision = tp / (tp + fp + 1e-6)
        # self.recall = tp / (tp + fn + 1e-6)


class ImportGraph():
    def __init__(self, save_dir,emb_mat,token2idx,q_rnn,q_mask):
        self.save_dir = save_dir
        self.emb_mat = emb_mat
        self.token2idx = token2idx
        self.q_rnn = q_rnn
        self.q_mask = q_mask
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        # sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.session_conf,graph=self.graph)

        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            self.model_yunnan = Model(flags.FLAGS, None, self.emb_mat, trainable=False, demo=True)
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            # saver.restore(sess, tf.train.latest_checkpoint(save_dir))
            # saver.restore(sess, os.path.join(save_dir, 'model_{}.ckpt'.format(43830)))
            saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))

    def get_pred(self,context):
        keras.backend.clear_session()  # 清除原本数据
        config = flags.FLAGS
        cs = convert_to_features(config, context, self.token2idx)
        pred = np.array
        for i, c in enumerate(cs):
            fd = {'context:0': np.expand_dims(c, 0),
                  'pre_query:0': self.q_rnn,
                  'pre_query_mask:0': self.q_mask,
                  'batch_size:0': 1}
            p = self.sess.run([self.model_yunnan.pred_probability], feed_dict=fd)
            if i == 0:
                pred = p
            else:
                pred = np.concatenate((pred, p), axis=0)
        return pred


class GET_VAL():
    def __init__(self):
        #####################定义结论模型路径#####################################################################
        BASE_DIR_jielun = './model_save/jielun'
        emb_mat_file_jielun = ("./prepro_/pretrained/joint_joint/02/emb_mat.json")
        emb_mat_file_jielun = BASE_DIR_jielun + emb_mat_file_jielun
        token2idx_file_jielun = ("./prepro_/pretrained/joint_joint/02/token2idx.json")
        token2idx_file_jielun = BASE_DIR_jielun + '/' + token2idx_file_jielun
        save_dir_jielun = ("./results/joint_jointdeploy/model/02/")
        save_dir_jielun = BASE_DIR_jielun + '/' + save_dir_jielun

        with open(BASE_DIR_jielun + '/q.pkl', 'rb') as f:
            q_rnn_jielun = pickle.load(f)
        q_rnn_jielun = np.array(q_rnn_jielun, dtype=np.float32)
        with open(BASE_DIR_jielun + '/q_mask.pkl', 'rb') as f:
            q_mask_jielun = pickle.load(f)
        q_mask_jielun = np.array(q_mask_jielun, dtype=np.bool)

        with open(emb_mat_file_jielun, 'r') as fh:
            emb_mat_jielun = np.array(json.load(fh), dtype=np.float32)
        with open(token2idx_file_jielun, 'r') as fh:
            token2idx_jielun = json.load(fh)
        #####################定义描述模型路径#####################################################################
        BASE_DIR_miaoshu = './model_save/miaoshu'
        emb_mat_file_miaoshu = ("./prepro_/pretrained/joint_joint/02/emb_mat.json")
        emb_mat_file_miaoshu = BASE_DIR_miaoshu + emb_mat_file_miaoshu
        token2idx_file_miaoshu = ("./prepro_/pretrained/joint_joint/02/token2idx.json")
        token2idx_file_miaoshu = BASE_DIR_miaoshu + '/' + token2idx_file_miaoshu
        save_dir_miaoshu = ("./results/joint_jointdeploy/model/02/")
        save_dir_miaoshu = BASE_DIR_miaoshu + '/' + save_dir_miaoshu
        with open(BASE_DIR_miaoshu + '/q.pkl', 'rb') as f:
            q_rnn_miaoshu = pickle.load(f)
        q_rnn_miaoshu = np.array(q_rnn_miaoshu, dtype=np.float32)
        with open(BASE_DIR_miaoshu + '/q_mask.pkl', 'rb') as f:
            q_mask_miaoshu = pickle.load(f)
        q_mask_miaoshu = np.array(q_mask_miaoshu, dtype=np.bool)
        with open(emb_mat_file_miaoshu, 'r') as fh:
            emb_mat_miaoshu = np.array(json.load(fh), dtype=np.float32)
        with open(token2idx_file_miaoshu, 'r') as fh:
            token2idx_miaoshu = json.load(fh)
        #####################这是一条分割线#####################################################################

        self.model_jielun = ImportGraph(save_dir_jielun, emb_mat_jielun, token2idx_jielun, q_rnn_jielun, q_mask_jielun)

        self.model_miaoshu = ImportGraph(save_dir_miaoshu, emb_mat_miaoshu, token2idx_miaoshu, q_rnn_miaoshu, q_mask_miaoshu)



    def get_miaoshu_pred_list(self,context):
        if context != '':
            try:
                pred = np.array(self.model_miaoshu.get_pred(context)) > 0.5
                pred = np.sum(np.array(pred).reshape([pred.shape[0], 10]), 0)
            except:
                import logging
                logging.warning('出错了。详细见：,%s' % str(context))
                pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            import logging
            logging.warning('context为空？--> %s ，已经重置pred为0' % str(context))
            pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        pred_list = []
        for i in range(len(pred)):
            if pred[i] >= 1:
                pred_list.append(1)
            else:
                pred_list.append(0)
        return pred_list

    def get_jielun_pred_list(self,context):
        if context != '':
            try:
                pred = np.array(self.model_jielun.get_pred(context)) > 0.5
                pred = np.sum(np.array(pred).reshape([pred.shape[0], 10]), 0)
            except:
                import logging
                logging.warning('出错了。详细见：,%s' % str(context))
                pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            import logging
            logging.warning('context为空？--> %s ，已经重置pred为0' % str(context))
            pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        pred_list = []
        for i in range(len(pred)):
            if pred[i] >= 1:
                pred_list.append(1)
            else:
                pred_list.append(0)
        return pred_list

    def run(self,miaoshu,jielun):
        miaoshu_pred = self.get_miaoshu_pred_list(miaoshu)
        jielun_pred = self.get_miaoshu_pred_list(jielun)

        sick_list = [
            '慢支炎',
            '肺气肿',
            '肺动脉高压',
            '肺部感染',
            '纤维灶',
            '支气管',
            '间质肺',
            '主动脉硬化',
            '空洞/空腔',
            '肺结节影',
        ]
        miaoshu_result = {}
        for i in range(len(miaoshu_pred)):
            if miaoshu_pred[i] >= 1:
                miaoshu_result[sick_list[i]] = 1
            else:
                miaoshu_result[sick_list[i]] = 0

        jielun_result = {}
        for i in range(len(jielun_pred)):
            if jielun_pred[i] >= 1:
                jielun_result[sick_list[i]] = 1
            else:
                jielun_result[sick_list[i]] = 0

        result = {}
        result['慢支炎'] = jielun_result['慢支炎']
        result['肺气肿'] = jielun_result['肺气肿']
        result['肺动脉高压'] = jielun_result['肺动脉高压']
        result['肺部感染'] = jielun_result['肺部感染']
        result['支气管'] = jielun_result['支气管']
        result['间质肺'] = jielun_result['间质肺']
        result['主动脉硬化'] = jielun_result['主动脉硬化']

        if jielun_result['纤维灶'] or miaoshu_result['纤维灶']:
            result['纤维灶'] = 1
        else:
            result['纤维灶'] = 0

        if jielun_result['空洞/空腔'] or miaoshu_result['空洞/空腔']:
            result['空洞/空腔'] = 1
        else:
            result['空洞/空腔'] = 0

        if jielun_result['肺结节影'] or miaoshu_result['肺结节影']:
            result['肺结节影'] = 1
        else:
            result['肺结节影'] = 0
        return result


if __name__ == '__main__':
    miaoshu = """ 1、双肺散在斑片状密度增高影，考虑为感染性病变。
    2、慢性支气管炎表现。
    3、左心室增大、肺动脉增宽。
    4、纵隔淋巴结显示。
    5、双侧胸膜腔少量积液，双侧胸膜增厚、粘连。
    6、扫描层面显示胆囊炎、胆结石。
    1、双肺散在斑片状密度增高影，考虑为感染性病变。
    2、慢性支气管炎表现。
    3、左心室增大、肺动脉增宽。
    4、纵隔淋巴结显示。
    5、双侧胸膜腔少量积液，双侧胸膜增厚、粘连。
    6、扫描层面显示胆囊炎、胆结石。"""
    jielun = """ 1、双肺散在斑片状密度增高影，考虑为感染性病变。
    2、慢性支气管炎表现。
    3、左心室增大、肺动脉增宽。
    4、纵隔淋巴结显示。
    5、双侧胸膜腔少量积液，双侧胸膜增厚、粘连。
    6、扫描层面显示胆囊炎、胆结石。"""

    demo = GET_VAL()
    print(demo.run(miaoshu, jielun))

    def to_excel(filename,outname):
        data = pd.read_excel(filename)
        for i in trange(len(data)):
            miaoshu = str(data['REPORTDESCRIBE'][i])
            jielun = str(data['REPORTDIAGNOSE'][i])
            result = demo.run(miaoshu,jielun)
            data['慢支炎'][i] = result['慢支炎']
            data['肺气肿'][i] = result['肺气肿']
            data['肺动脉高压'][i] = result['肺动脉高压']
            data['肺部感染'][i] = result['肺部感染']
            data['支气管'][i] = result['支气管']
            data['间质肺'][i] = result['间质肺']
            data['主动脉硬化'][i] = result['主动脉硬化']
            data['纤维灶'][i] = result['纤维灶']
            data['空洞/空腔'][i] = result['空洞/空腔']
            data['肺结节影'][i] = result['肺结节影']
            data['result'][i] = str(result)

        data.to_excel(outname)

    to_excel('./test/征象错误集.xlsx','./out/征象错误集_out.xlsx')
