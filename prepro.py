# -*-coding:utf-8-*-
import os
import xlrd
import ujson as json
from tqdm import tqdm
from itertools import islice
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold


def process_file(filename):
    data = xlrd.open_workbook(filename)
    diagnoses = data.sheets()[1]
    dia_xs = diagnoses.col_values(0)[1:]
    dia_ys = []
    for i in range(10):
        dia_ys.append(diagnoses.col_values(i + 1)[1:])
    dia_ys = list(zip(*dia_ys))
    descriptions = data.sheets()[2]
    des_xs = descriptions.col_values(0)[1:]
    des_ys = []
    for i in range(10):
        des_ys.append(descriptions.col_values(i + 1)[1:])
    des_ys = list(zip(*des_ys))

    assert len(dia_xs) == len(dia_ys)
    assert len(des_xs) == len(des_ys)
    assert len(dia_ys[0]) == 10
    assert len(des_ys[0]) == 10

    # dia_xs = dia_xs[:450]
    # dia_ys = dia_ys[:450]
    # des_xs = des_xs[:1037]
    # des_ys = des_ys[:1037]

    def _prepro(xs, ys):
        xs_clean = []
        ys_clean = []
        for i in range(len(xs)):
            if xs[i][:2] == 'ZY':
                continue
            elif xs[i][0] in '0123456789' and xs[i][1] in '.、':
                xs_clean.append(xs[i][2:])
            elif xs[i][0] in '0123456789' and xs[i][1] in '0123456789' and xs[i][2] in '.、':
                xs_clean.append(xs[i][3:])
            else:
                xs_clean.append(xs[i])
            ys_clean.append([1. if y else 0. for y in ys[i]])
        return xs_clean, ys_clean

    dia_xs, dia_ys = _prepro(dia_xs, dia_ys)
    des_xs, des_ys = _prepro(des_xs, des_ys)

    # print(dia_xs[-10:])
    # print(dia_ys[-10:])
    # print(des_xs[-10:])
    # print(des_ys[-10:])

    return dia_xs, dia_ys, des_xs, des_ys


def build_examples(xs, q, ys):
    examples = []
    for i in range(len(xs)):
        batch_example = {'x': xs[i], 'q': q, 'y': ys[i]}
        examples.append(batch_example)
    counter = Counter()
    for x in xs:
        for char in x:
            counter[char] += 1
    for char in q:
        counter[char] += len(xs)
    return examples, counter


def get_embedding(counter, limit=-1, emb_file=None, vec_size=None):
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        with open(emb_file, "rb") as fh:
            for line_b in tqdm(islice(fh, 1, None)):
                try:
                    line_u = line_b.decode('utf-8')
                except Exception:
                    continue
                array = line_u.split(' ')
                word = array[0]
                vector = [float(v) for v in array[1:-1]]
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
            vec_size = len(vector)
        print("vec_size:", vec_size)
        print("{} / {} tokens have corresponding embedding vector".format(
            len(embedding_dict), len(filtered_elements)))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                                     token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]  # !!OOV不应全0初始化!!
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    idx2token_dict = {idx: token for token, idx in token2idx_dict.items()}
    return emb_mat, token2idx_dict, idx2token_dict


def build_features(config, examples, data_type, out_file, char2idx_dict):
    x_limit = config.x_limit
    q_limit = config.q_limit

    def _filter_func(example):
        return len(example['x']) > x_limit or len(example['q']) > q_limit

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples, ncols=100):
        total_ += 1

        if _filter_func(example):
            continue

        total += 1
        x_idxs = np.zeros([x_limit], dtype=np.int32)
        q_idxs = np.zeros([q_limit], dtype=np.int32)

        for i, char in enumerate(example['x']):
            x_idxs[i] = _get_char(char)

        for i, char in enumerate(example['q']):
            q_idxs[i] = _get_char(char)

        record = tf.train.Example(features=tf.train.Features(feature={
            "x_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_idxs.tostring()])),
            "q_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[q_idxs.tostring()])),
            "y": tf.train.Feature(float_list=tf.train.FloatList(value=[example['y']])),
        }))
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    writer.close()
    return total


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    """joint all prepro with cross-validation"""
    emb_file = config.init_emb_file if config.pretrained_emb else None
    vec_size = config.emb_dim
    dia_xs, dia_ys, des_xs, des_ys = process_file(config.data_file)
    xs = dia_xs + des_xs
    ys = dia_ys + des_ys
    xs = np.array(xs)
    ys = np.array(list(zip(*ys)))
    assert ys.shape[0] == 10

    qs = ['慢支炎肺纹理增多增粗紊乱',
          '肺气肿透亮度增加膈肌低平肺大泡',
          '肺动脉高压右下肺动脉增宽肺动脉段突出右室增大',
          '肺部感染单发多发斑片状阴影',
          '陈旧性肺结核纤维条索影',
          '支气管扩张卷发状阴影囊状透光区环形阴影轨道征',
          '间质肺磨玻璃网格状蜂窝状阴影',
          '主动脉弓硬化',
          '空洞空腔',
          '肺结节影']

    meta = {}
    train_examples = {}
    valid_examples = {}
    test_examples = []
    counter = Counter()
    skf = StratifiedKFold(config.cv_k, shuffle=False)
    for y_type in range(10):
        xs_, xs_test, ys_, ys_test = train_test_split(xs, ys[y_type], test_size=.2, stratify=ys[y_type])
        if config.deploy:
            xs_, ys_ = xs, ys[y_type]
        q = qs[y_type]
        for x in xs_:
            for char in x:
                counter[char] += 1
        for char in q:
            counter[char] += len(xs_)
        # test set
        examples, _ = build_examples(xs_test, q, ys_test)
        test_examples += examples
        i = 0
        for train_index, valid_index in skf.split(xs_, ys_):
            xs_train, xs_valid = xs_[train_index], xs_[valid_index]
            ys_train, ys_valid = ys_[train_index], ys_[valid_index]
            if config.deploy:
                xs_train = np.concatenate((xs_train, xs_valid), 0)
                ys_train = np.concatenate((ys_train, ys_valid), 0)
            # train set
            examples, _ = build_examples(xs_train, q, ys_train)
            if i in train_examples:
                train_examples[i] += examples
            else:
                train_examples[i] = examples
            # valid set
            examples, _ = build_examples(xs_valid, q, ys_valid)
            if i in valid_examples:
                valid_examples[i] += examples
            else:
                valid_examples[i] = examples
            i += 1
    emb_mat, token2idx_dict, idx2token_dict = get_embedding(counter, emb_file=emb_file, vec_size=vec_size)
    out_dir = os.path.join(config.prepro_home, 'joint_joint')
    assert len(train_examples) == len(valid_examples) == config.cv_k
    for i in range(config.cv_k):
        out_cv_dir = os.path.join(out_dir, '{:0>2d}'.format(i + 1))
        if not os.path.exists(out_cv_dir):
            os.makedirs(out_cv_dir)
        print('-' * 10 + 'cv-{:0>2d}'.format(i + 1) + '-' * 10)
        # train set
        out_file = os.path.join(out_cv_dir, "train.tfrecords")
        train_total = build_features(config, train_examples[i], 'train', out_file, token2idx_dict)
        # valid set
        out_file = os.path.join(out_cv_dir, "valid.tfrecords")
        valid_total = build_features(config, valid_examples[i], 'valid', out_file, token2idx_dict)
        # test set
        out_file = os.path.join(out_cv_dir, "test.tfrecords")
        test_total = build_features(config, test_examples, 'test', out_file, token2idx_dict)
        meta = {'train_total': train_total, 'valid_total': valid_total, 'test_total': test_total}
        save(os.path.join(out_cv_dir, "emb_mat.json"), emb_mat, message="embedding matrix")
        save(os.path.join(out_cv_dir, "meta.json"), meta, message="meta")
        save(os.path.join(out_cv_dir, "token2idx.json"), token2idx_dict, message="token2idx dict")
        save(os.path.join(out_cv_dir, "idx2token.json"), idx2token_dict, message="idx2token dict")


def convert_to_features(config, context, token2idx):
    context = context.replace('\r', '').replace('\n', '').replace(' ', '').strip()
    qs = ['慢支炎肺纹理增多增粗紊乱',
          '肺气肿透亮度增加膈肌低平肺大泡',
          '肺动脉高压右下肺动脉增宽肺动脉段突出右室增大',
          '肺部感染单发多发斑片状阴影',
          '陈旧性肺结核纤维条索影',
          '支气管扩张卷发状阴影囊状透光区环形阴影轨道征',
          '间质肺磨玻璃网格状蜂窝状阴影',
          '主动脉弓硬化',
          '空洞空腔',
          '肺结节影']

    def cut(passage):
        cutset = set('。！？!?：:;；')
        sentences = []
        s = 0
        for i, char in enumerate(passage):
            if char in cutset:
                sentences.append(passage[s:i + 1])
                s = i + 1
        if s != len(passage):
            sentences.append(passage[s:])
        return sentences

    x_limit = config.test_x_limit
    q_limit = config.test_q_limit

    context = cut(context)

    def _get_char(char):
        if char in token2idx:
            return token2idx[char]
        return 1

    context_num = len(context)
    x_idxs = np.zeros([context_num, x_limit], dtype=np.int32)
    for i in range(context_num):
        for j, char in enumerate(context[i][:x_limit]):
            x_idxs[i][j] = _get_char(char)

    return x_idxs
