import tensorflow as tf


def get_record_parser(config):
    def parse(example):
        x_limit = config.x_limit
        q_limit = config.q_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "x_idxs": tf.FixedLenFeature([], tf.string),
                                               "q_idxs": tf.FixedLenFeature([], tf.string),
                                               "y": tf.FixedLenFeature([], tf.float32)
                                           })
        x_idxs = tf.reshape(tf.decode_raw(
            features["x_idxs"], tf.int32), [x_limit])
        q_idxs = tf.reshape(tf.decode_raw(
            features["q_idxs"], tf.int32), [q_limit])
        y = features["y"]
        return x_idxs, q_idxs, y

    return parse


def get_batch_dataset(record_file, parser, total, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(total).repeat().batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset
