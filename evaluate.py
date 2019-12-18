import os

import tensorflow as tf

from Base.data_loader import DataLoader
from Config.hyperparams import HyperParams
from net import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

hp = HyperParams()
loader = DataLoader(hp, mode='test')

tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    n = Net(hp=hp)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(hp.logdir, 'epoch3.ckpt'))

    index = 0
    total_loss = 0
    total_pAt1 = 0
    total_pAt5 = 0
    total_mrr = 0
    total_meanRank = 0
    print('batch size:', hp.Data.batch_size)

    for VGG, C3D, HIS, QUE, ANS, CAN, ANS_ID in loader.get_batch_data():
        mean_loss, pAt1, pAt5, mrr, meanRank = sess.run(
            (n.mean_loss, n.pAt1, n.pAt5, n.mrr, n.meanRank), feed_dict={
                n.VGG: VGG,
                n.C3D: C3D,
                n.HIS: HIS,
                n.QUE: QUE,
                n.ANS: ANS,
                n.CAN: CAN,
                n.ANS_ID: ANS_ID,
            })

        total_loss += mean_loss
        total_pAt1 += pAt1
        total_pAt5 += pAt5
        total_mrr += mrr
        total_meanRank += meanRank
        index += 1

        if index % 100 == 1:
            print('index: ', index,
                  'accuracy:', pAt1,
                  'loss:', total_loss / index,
                  'p@1:', total_pAt1 / index,
                  'p@5:', total_pAt5 / index,
                  'mrr:', total_mrr / index,
                  'meanRank:', total_meanRank / index)

    print('summary')
    print('loss:', total_loss / index)
    print('accuracy:', total_pAt1 / index)
    print('loss:', total_loss / index)
    print('p@1:', total_pAt1 / index)
    print('p@5:', total_pAt5 / index)
    print('mrr:', total_mrr / index)
    print('meanRank:', total_meanRank / index)
