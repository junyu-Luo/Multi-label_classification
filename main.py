# -*-coding:utf-8-*-
import os
import ujson as json
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from util import get_record_parser, get_batch_dataset, get_dataset
from prepro import convert_to_features
from model import Model


def train(config):
    with open(config.emb_mat_file, 'r') as fh:
        emb_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.meta_file, 'r') as fh:
        meta = json.load(fh)

    train_total = meta["train_total"]
    test_total = meta["test_total"]

    print("Building model...")
    parser = get_record_parser(config)

    train_dataset = get_batch_dataset(config.train_record_file, parser, train_total, config)
    test_dataset = get_dataset(config.test_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()

    model = Model(config, iterator, emb_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    lr = config.learning_rate

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        global_step = max(sess.run(model.global_step), 0)

        step_per_epoch = train_total // config.batch_size
        test_step_per_epoch = test_total // config.batch_size
        epoch = global_step // step_per_epoch + 1
        epoch_step = global_step % step_per_epoch
        print("init epoch: {}, init epoch_step: {}".format(epoch, epoch_step))
        for epoch in tqdm(range(epoch, config.epoches + 1), ncols=100):
            # for epoch in range(epoch, config.epoches + 1):
            losses, accs, tps, fps, fns = 0., 0., 0., 0., 0.
            for _ in tqdm(range(global_step % step_per_epoch, step_per_epoch), ncols=100):
                # for _ in range(global_step % step_per_epoch, step_per_epoch):
                global_step = sess.run(model.global_step) + 1
                loss, acc, tp, fp, fn, _ = sess.run(
                    [model.loss, model.acc, model.tp, model.fp, model.fn, model.train_op],
                    feed_dict={handle: train_handle, model.dropout: config.dropout, model.lr: lr})
                losses += loss
                accs += acc
                tps += tp
                fps += fp
                fns += fn
                if global_step % 100 == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)

            loss = losses / step_per_epoch
            acc = accs / step_per_epoch
            precision = tps / (tps + fps + 1e-6)
            recall = tps / (tps + fns + 1e-6)
            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="train/loss", simple_value=loss), ])
            writer.add_summary(loss_sum, epoch)
            acc_sum = tf.Summary(value=[tf.Summary.Value(tag="train/acc", simple_value=acc), ])
            writer.add_summary(acc_sum, epoch)
            precision_sum = tf.Summary(value=[tf.Summary.Value(tag="train/precision", simple_value=precision), ])
            writer.add_summary(precision_sum, epoch)
            recall_sum = tf.Summary(value=[tf.Summary.Value(tag="train/recall", simple_value=recall), ])
            writer.add_summary(recall_sum, epoch)
            print()
            print()
            print('tps:', tps)
            print('fps:', fps)
            print('fns:', fns)
            print('TRAIN Epoch {}: loss {:.4f} acc {:.4f} precision {:.4f} recall {:.4f}'.format(
                epoch, loss, acc, precision, recall))

            losses, accs, tps, fps, fns = 0., 0., 0., 0., 0.
            # for _ in tqdm(range(test_step_per_epoch), ncols=100):
            for _ in range(test_step_per_epoch):
                loss, acc, tp, fp, fn = sess.run(
                    [model.loss, model.acc, model.tp, model.fp, model.fn],
                    feed_dict={handle: test_handle})
                losses += loss
                accs += acc
                tps += tp
                fps += fp
                fns += fn
            loss = losses / test_step_per_epoch
            acc = accs / test_step_per_epoch
            precision = tps / (tps + fps + 1e-6)
            recall = tps / (tps + fns + 1e-6)
            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="test/loss", simple_value=loss), ])
            writer.add_summary(loss_sum, epoch)
            acc_sum = tf.Summary(value=[tf.Summary.Value(tag="test/acc", simple_value=acc), ])
            writer.add_summary(acc_sum, epoch)
            precision_sum = tf.Summary(value=[tf.Summary.Value(tag="test/precision", simple_value=precision), ])
            writer.add_summary(precision_sum, epoch)
            recall_sum = tf.Summary(value=[tf.Summary.Value(tag="test/recall", simple_value=recall), ])
            writer.add_summary(recall_sum, epoch)
            print('tps:', tps)
            print('fps:', fps)
            print('fns:', fns)
            print('TEST  Epoch {}: loss {:.4f} acc {:.4f} precision {:.4f} recall {:.4f}'.format(
                epoch, loss, acc, precision, recall))

        # if epoch % 20 == 0:
        # 	filename = os.path.join(
        # 		config.save_dir, "model_{}.ckpt".format(global_step))
        # 	saver.save(sess, filename)


def cv_train(config):
    def _train(config, k):
        print('-' * 10 + 'cv-{:0>2d}'.format(k) + '-' * 10)
        prepro_home = os.path.join(config.prepro_dir, '{:0>2d}'.format(k))
        emb_mat_file = os.path.join(prepro_home, 'emb_mat.json')
        meta_file = os.path.join(prepro_home, 'meta.json')
        train_record_file = os.path.join(prepro_home, 'train.tfrecords')
        valid_record_file = os.path.join(prepro_home, 'valid.tfrecords')
        test_record_file = os.path.join(prepro_home, 'test.tfrecords')
        log_dir = os.path.join(config.log_dir, '{:0>2d}'.format(k))
        save_dir = os.path.join(config.save_dir, '{:0>2d}'.format(k))
        answer_dir = os.path.join(config.answer_dir, '{:0>2d}'.format(k))

        with open(emb_mat_file, 'r') as fh:
            emb_mat = np.array(json.load(fh), dtype=np.float32)
        with open(meta_file, 'r') as fh:
            meta = json.load(fh)

        train_total = meta["train_total"]
        valid_total = meta["valid_total"]
        test_total = meta["test_total"]

        print("Building model...")
        parser = get_record_parser(config)

        graph = tf.Graph()
        with graph.as_default() as g:
            train_dataset = get_batch_dataset(train_record_file, parser, train_total, config)
            valid_dataset = get_dataset(valid_record_file, parser, config)
            test_dataset = get_dataset(test_record_file, parser, config)
            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                handle, train_dataset.output_types, train_dataset.output_shapes)
            train_iterator = train_dataset.make_one_shot_iterator()
            valid_iterator = valid_dataset.make_one_shot_iterator()
            test_iterator = test_dataset.make_one_shot_iterator()

            model = Model(config, iterator, emb_mat)

            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

            lr = config.learning_rate

            with tf.Session(config=sess_config) as sess:
                writer = tf.summary.FileWriter(log_dir)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                train_handle = sess.run(train_iterator.string_handle())
                valid_handle = sess.run(valid_iterator.string_handle())
                test_handle = sess.run(test_iterator.string_handle())
                if os.path.exists(os.path.join(save_dir, "checkpoint")):
                    saver.restore(sess, tf.train.latest_checkpoint(save_dir))
                global_step = max(sess.run(model.global_step), 0)

                step_per_epoch = train_total // config.batch_size
                valid_step_per_epoch = valid_total // config.batch_size
                test_step_per_epoch = test_total // config.batch_size
                epoch = global_step // step_per_epoch + 1
                epoch_step = global_step % step_per_epoch
                print("init epoch: {}, init epoch_step: {}".format(epoch, epoch_step))
                for epoch in tqdm(range(epoch, config.epoches + 1), ncols=100):
                    # for epoch in range(epoch, config.epoches + 1):
                    losses, accs, tps, fps, fns = 0., 0., 0., 0., 0.
                    for _ in tqdm(range(global_step % step_per_epoch, step_per_epoch), ncols=100):
                        # for _ in range(global_step % step_per_epoch, step_per_epoch):
                        global_step = sess.run(model.global_step) + 1
                        loss, acc, tp, fp, fn, _ = sess.run(
                            [model.loss, model.acc, model.tp, model.fp, model.fn, model.train_op],
                            feed_dict={handle: train_handle, model.dropout: config.dropout, model.lr: lr})
                        losses += loss
                        accs += acc
                        tps += tp
                        fps += fp
                        fns += fn
                        if global_step % 100 == 0:
                            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                            writer.add_summary(loss_sum, global_step)

                    loss = losses / step_per_epoch
                    acc = accs / step_per_epoch
                    precision = tps / (tps + fps + 1e-6)
                    recall = tps / (tps + fns + 1e-6)
                    f1 = 2 * precision * recall / (precision + recall + 1e-6)
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="train/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, epoch)
                    acc_sum = tf.Summary(value=[tf.Summary.Value(tag="train/acc", simple_value=acc), ])
                    writer.add_summary(acc_sum, epoch)
                    precision_sum = tf.Summary(
                        value=[tf.Summary.Value(tag="train/precision", simple_value=precision), ])
                    writer.add_summary(precision_sum, epoch)
                    recall_sum = tf.Summary(value=[tf.Summary.Value(tag="train/recall", simple_value=recall), ])
                    writer.add_summary(recall_sum, epoch)
                    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="train/f1", simple_value=f1), ])
                    writer.add_summary(f1_sum, epoch)
                    print()
                    print()
                    print('tps:', tps)
                    print('fps:', fps)
                    print('fns:', fns)
                    print('TRAIN Epoch {}: loss {:.4f} acc {:.4f} precision {:.4f} recall {:.4f} f1 {:.4f}'.format(
                        epoch, loss, acc, precision, recall, f1))

                    losses, accs, tps, fps, fns = 0., 0., 0., 0., 0.
                    # for _ in tqdm(range(valid_step_per_epoch), ncols=100):
                    for _ in range(valid_step_per_epoch):
                        loss, acc, tp, fp, fn = sess.run(
                            [model.loss, model.acc, model.tp, model.fp, model.fn],
                            feed_dict={handle: valid_handle})
                        losses += loss
                        accs += acc
                        tps += tp
                        fps += fp
                        fns += fn
                    loss = losses / valid_step_per_epoch
                    acc = accs / valid_step_per_epoch
                    precision = tps / (tps + fps + 1e-6)
                    recall = tps / (tps + fns + 1e-6)
                    f1 = 2 * precision * recall / (precision + recall + 1e-6)
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="valid/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, epoch)
                    acc_sum = tf.Summary(value=[tf.Summary.Value(tag="valid/acc", simple_value=acc), ])
                    writer.add_summary(acc_sum, epoch)
                    precision_sum = tf.Summary(
                        value=[tf.Summary.Value(tag="valid/precision", simple_value=precision), ])
                    writer.add_summary(precision_sum, epoch)
                    recall_sum = tf.Summary(value=[tf.Summary.Value(tag="valid/recall", simple_value=recall), ])
                    writer.add_summary(recall_sum, epoch)
                    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="valid/f1", simple_value=f1), ])
                    writer.add_summary(f1_sum, epoch)
                    print('tps:', tps)
                    print('fps:', fps)
                    print('fns:', fns)
                    print('VALID Epoch {}: loss {:.4f} acc {:.4f} precision {:.4f} recall {:.4f} f1 {:.4f}'.format(
                        epoch, loss, acc, precision, recall, f1))

                    losses, accs, tps, fps, fns = 0., 0., 0., 0., 0.
                    # for _ in tqdm(range(test_step_per_epoch), ncols=100):
                    for _ in range(test_step_per_epoch):
                        loss, acc, tp, fp, fn = sess.run(
                            [model.loss, model.acc, model.tp, model.fp, model.fn],
                            feed_dict={handle: test_handle})
                        losses += loss
                        accs += acc
                        tps += tp
                        fps += fp
                        fns += fn
                    loss = losses / test_step_per_epoch
                    acc = accs / test_step_per_epoch
                    precision = tps / (tps + fps + 1e-6)
                    recall = tps / (tps + fns + 1e-6)
                    f1 = 2 * precision * recall / (precision + recall + 1e-6)
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="test/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, epoch)
                    acc_sum = tf.Summary(value=[tf.Summary.Value(tag="test/acc", simple_value=acc), ])
                    writer.add_summary(acc_sum, epoch)
                    precision_sum = tf.Summary(value=[tf.Summary.Value(tag="test/precision", simple_value=precision), ])
                    writer.add_summary(precision_sum, epoch)
                    recall_sum = tf.Summary(value=[tf.Summary.Value(tag="test/recall", simple_value=recall), ])
                    writer.add_summary(recall_sum, epoch)
                    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="test/f1", simple_value=f1), ])
                    writer.add_summary(f1_sum, epoch)
                    print('tps:', tps)
                    print('fps:', fps)
                    print('fns:', fns)
                    print('TEST  Epoch {}: loss {:.4f} acc {:.4f} precision {:.4f} recall {:.4f} f1 {:.4f}'.format(
                        epoch, loss, acc, precision, recall, f1))

                    if epoch % 10 == 0:
                        filename = os.path.join(
                            save_dir, "model_{}.ckpt".format(global_step))
                        saver.save(sess, filename)

    if config.seq:
        for k in range(1, config.cv_k + 1):
            _train(config, k)
    else:
        _train(config, config.k)


def cv_test(config):
    def _test(config, k):
        print('-' * 10 + 'cv-{:0>2d}'.format(k) + '-' * 10)
        prepro_home = os.path.join(config.prepro_dir, '{:0>2d}'.format(k))
        emb_mat_file = os.path.join(prepro_home, 'emb_mat.json')
        meta_file = os.path.join(prepro_home, 'meta.json')
        idx2token_file = os.path.join(prepro_home, 'idx2token.json')
        test_record_file = os.path.join(prepro_home, 'test.tfrecords')
        log_dir = os.path.join(config.log_dir, 'ensemble')
        save_dir = os.path.join(config.save_dir, '{:0>2d}'.format(k))
        answer_dir = os.path.join(config.answer_dir, '{:0>2d}'.format(k))

        with open(emb_mat_file, 'r') as fh:
            emb_mat = np.array(json.load(fh), dtype=np.float32)
        with open(meta_file, 'r') as fh:
            meta = json.load(fh)
        with open(idx2token_file, 'r') as fh:
            idx2token = json.load(fh)

        test_total = meta["test_total"]

        print("Building model...")
        parser = get_record_parser(config)

        graph = tf.Graph()
        with graph.as_default() as g:
            test_dataset = get_dataset(test_record_file, parser, config)
            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                handle, test_dataset.output_types, test_dataset.output_shapes)
            test_iterator = test_dataset.make_one_shot_iterator()

            model = Model(config, iterator, emb_mat)

            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

            with tf.Session(config=sess_config) as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                test_handle = sess.run(test_iterator.string_handle())
                if os.path.exists(os.path.join(save_dir, "checkpoint")):
                    saver.restore(sess, tf.train.latest_checkpoint(save_dir))
                global_step = max(sess.run(model.global_step), 0)

                test_step_per_epoch = test_total // config.batch_size

                cs, qs, preds, labels = [], [], [], []
                accs, tps, fps, fns = 0., 0., 0., 0.
                for _ in tqdm(range(test_step_per_epoch), ncols=100):
                    # for _ in range(test_step_per_epoch):
                    c, q, y, pred, label, acc, tp, fp, fn = sess.run(
                        [model.c, model.q, model.y, model.pred, model.label,
                         model.acc, model.tp, model.fp, model.fn],
                        feed_dict={handle: test_handle})
                    for i in range(len(c)):
                        cs.append([idx2token[str(w)] for w in c[i] if w])
                        qs.append([idx2token[str(w)] for w in q[i] if w])
                    preds += list(pred)
                    labels += list(label)
                    accs += acc
                    tps += tp
                    fps += fp
                    fns += fn
                acc = accs / test_step_per_epoch
                precision = tps / (tps + fps + 1e-6)
                recall = tps / (tps + fns + 1e-6)
                print('tps:', tps)
                print('fps:', fps)
                print('fns:', fns)
                print('TEST  CV {} acc {:.4f} precision {:.4f} recall {:.4f}'.format(
                    k, acc, precision, recall))
        return cs, qs, preds, labels

    def _workbook_op(workbook, k, cs, qs, preds, labels):
        worksheet = workbook.add_sheet(k)
        worksheet.col(0).width = 256 * 100
        worksheet.col(1).width = 256 * 40
        worksheet.write(0, 0, label='passage')
        worksheet.write(0, 1, label='question')
        worksheet.write(0, 2, label='prediction')
        worksheet.write(0, 3, label='label')
        style = xlwt.easyxf('pattern: pattern solid, fore_colour red;')
        for i in range(len(cs)):
            worksheet.write(i + 1, 0, label=cs[i])
            worksheet.write(i + 1, 1, label=qs[i])
            if preds[i] == labels[i]:
                worksheet.write(i + 1, 2, label=int(preds[i]))
            else:
                worksheet.write(i + 1, 2, label=int(preds[i]), style=style)
            worksheet.write(i + 1, 3, label=int(labels[i]))

    import xlwt
    workbook = xlwt.Workbook(encoding='utf8')
    if config.seq:
        preds = []
        for k in range(1, config.cv_k + 1):
            cs, qs, pred, labels = _test(config, k)
            preds.append(pred)
            _workbook_op(workbook, '{:0>2d}'.format(k), cs, qs, pred, labels)
        mix_preds = np.sum(preds, 0) > config.cv_k // 2
        acc = np.equal(mix_preds, labels).mean()
        tps = np.sum(np.logical_and(mix_preds, labels))
        fps = np.sum(np.logical_and(mix_preds, np.logical_not(labels)))
        fns = np.sum(np.logical_and(np.logical_not(mix_preds), labels))
        precision = tps / (tps + fps + 1e-6)
        recall = tps / (tps + fns + 1e-6)
        print('tps:', tps)
        print('fps:', fps)
        print('fns:', fns)
        print('TEST  ENSEM acc {:.4f} precision {:.4f} recall {:.4f}'.format(
            acc, precision, recall))
        _workbook_op(workbook, 'ENSEMBLE', cs, qs, mix_preds, labels)
    else:
        cs, qs, preds, labels = _test(config, config.k)
        _workbook_op(workbook, '{:0>2d}'.format(config.k), cs, qs, preds, labels)
    workbook.save(os.path.join(config.answer_dir, 'results.xls'))


def demo(config):
    prepro_home = os.path.join(config.prepro_dir, '{:0>2d}'.format(config.k))
    emb_mat_file = os.path.join(prepro_home, 'emb_mat.json')
    token2idx_file = os.path.join(prepro_home, 'token2idx.json')
    save_dir = os.path.join(config.save_dir, '{:0>2d}'.format(config.k))
    with open(emb_mat_file, 'r') as fh:
        emb_mat = np.array(json.load(fh), dtype=np.float32)
    with open(token2idx_file, 'r') as fh:
        token2idx = json.load(fh)

    import bottle
    app = bottle.Bottle()

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    sess = tf.Session(config=sess_config)
    model = Model(config, None, emb_mat, trainable=False, demo=True)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    saver.restore(sess, os.path.join(save_dir, 'model_{}.ckpt'.format(43830)))

    @app.post('/')
    def answer():
        context = bottle.request.forms.get('context')
        c, q, context_num, batch_size = convert_to_features(config, context, token2idx)
        fd = {'context:0': c,
              'query:0': q,
              'batch_size:0': batch_size}
        pred = sess.run([model.pred], feed_dict=fd)
        pred = np.sum(np.array(pred).reshape([context_num, 10]), 0)
        pred = ''.join(pred.astype(bool).astype(int).astype(str))
        return pred

    app.run(port=8080, host='0.0.0.0')
