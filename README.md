# GRU+attention模型

## 训练过程

* 将训练数据处理成与 `./data/dataset.xlsx` 相同的规范格式
* 将 `config.py` 的第9行 `data_file = 'data/dataset.xlsx'` 改成新的训练数据文件路径
* 执行 `python config.py --mode prepro` 命令进行数据预处理，得到 `./prepro_/` 文件夹
* 执行 `python config.py --mode train` 命令进行训练，训练完毕后，得到 `./results/` 文件夹
* 执行 `python get_p.py` 命令，得到 `q.pkl` 和 `q_mask.pkl`

* 执行 `python predict.py` 进行预测

## 一些细节

* 训练过程为50个Epoch，训练时可用tensorboard实时查看指标变化，方法为在项目目录下执行`tensorboard --logdir=./results`
* `config.py`中有个`deploy`参数（第68行），默认为`True`，此时将用全部数据训练，为了与一般情况兼容，此时仍然会从数据中抽样出验证集和测试集，但此时验证集和测试集已包含在训练数据中，因此这两个数据集上的指标不再具有原来的意义，本质变成了训练集上的指标。

## Requirements

* Tensorflow 1.8.0
* scikit-learn
* numpy
* ujson
* tqdm
* xlrd
* six