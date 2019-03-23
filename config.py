import os
import tensorflow as tf

from prepro import prepro
from main import train, cv_train, cv_test, demo


flags = tf.flags

data_file = 'data/dataset.xlsx'
init_emb_file = 'data/myembd(all100dim).txt'

x_type = 'joint'
y_type = 'joint'
cv_k = 2
pretrained_emb = True
trainable_emb = False
if not pretrained_emb:
    trainable_emb = True

prepro_home = 'prepro_'
if pretrained_emb:
    prepro_home = os.path.join(prepro_home, 'pretrained')
else:
    prepro_home = os.path.join(prepro_home, 'random')
prepro_dir = os.path.join(prepro_home, '{}_{}'.format(x_type, y_type))
train_record_file = os.path.join(prepro_dir, "train.tfrecords")
test_record_file = os.path.join(prepro_dir, "test.tfrecords")
emb_mat_file = os.path.join(prepro_dir, "emb_mat.json")
meta_file = os.path.join(prepro_dir, "meta.json")

model_name = "deploy"
model_name = '{}_{}'.format(x_type, y_type) + model_name
# model_name += "_debug"
model_dir = os.path.join("results", model_name)
log_dir = os.path.join(model_dir, "event")
save_dir = os.path.join(model_dir, "model")
answer_dir = os.path.join(model_dir, "answer")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")

flags.DEFINE_string("x_type", x_type, "Type of x, dia, des or joint")
flags.DEFINE_string("y_type", y_type, "Type of y, from 0-9 or joint")
flags.DEFINE_integer("cv_k", cv_k, "k fold for cross validation")

flags.DEFINE_string("data_file", data_file, "Data source file")
flags.DEFINE_string("init_emb_file", init_emb_file, "Initial embedding file")
flags.DEFINE_boolean("pretrained_emb", pretrained_emb, "Whether to use pretrained embedding")
flags.DEFINE_boolean("trainable_emb", trainable_emb, "Whether to train pretrained embedding")

flags.DEFINE_string("prepro_home", prepro_home, "Directory for prepro home")
flags.DEFINE_string("prepro_dir", prepro_dir, "Directory for prepro data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("answer_dir", answer_dir, "Directory for output answers")

flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("emb_mat_file", emb_mat_file, "Out file for embedding matrix")
flags.DEFINE_string("meta_file", meta_file, "Out file for meta")

flags.DEFINE_boolean("deploy", True, "If True, use whole data to train.")
flags.DEFINE_boolean("cudnn", False, "Whether to use cudnn rnn cell")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("seq", False, "Whether to train/test each cv model sequentially")
flags.DEFINE_integer("k", 2, "Which cv data to use, start from 1")

flags.DEFINE_integer("x_limit", 100, "Limit length for paragraph")
flags.DEFINE_integer("q_limit", 22, "Limit length for question")
flags.DEFINE_integer("test_x_limit", 200, "Limit length for paragraph")
flags.DEFINE_integer("test_q_limit", 22, "Limit length for question")

flags.DEFINE_integer("char_count_limit", -1, "Min count for char")
flags.DEFINE_integer("emb_dim", 100, "Embedding dimension")

flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("epoches", 50, "Number of epoches")
flags.DEFINE_integer("hidden", 100, "Hidden size")
flags.DEFINE_integer("num_layers", 1, "Hidden size")
flags.DEFINE_boolean("birnn", False, "Whether to use birnn")
flags.DEFINE_boolean("bidaf", True, "Whether to use bidaf")
flags.DEFINE_float("learning_rate", 1., "Learning rate")
flags.DEFINE_float("dropout", 0.2, "dropout rate")
flags.DEFINE_boolean("grad_clip_flag", True, "Whether to clip gradient")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
# flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("decay", 0., "Exponential moving average decay")
# flags.DEFINE_float("l2_norm", 1e-5, "L2 norm scale")
flags.DEFINE_float("l2_norm", 0., "L2 norm scale")
flags.DEFINE_float("gamma", 2., "gamma of focal loss")


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        # train(config)
        cv_train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "test":
        cv_test(config)
    elif config.mode == "demo":
        demo(config)


if __name__ == "__main__":
    tf.app.run()
