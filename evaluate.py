import os

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from Base.data_loader import DataLoader
from Config.hyperparams import HyperParams
from net import Net

hp = HyperParams()
loader = DataLoader(hp, mode='test')

tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    n = Net(hp=hp)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(hp.logdir, 'epoch0.ckpt'))

    index = 0
    total_loss = 0
    total_accuracy = 0
    print('batch size:', hp.Data.batch_size)

    for VGG, C3D, HIS, QUE, ANS, CAN, ANS_ID, turn_nums in loader.get_batch_data():
        mean_loss, accuracy = sess.run((n.mean_loss, n.accuracy), feed_dict={
            n.VGG: VGG,
            n.C3D: C3D,
            n.HIS: HIS,
            n.QUE: QUE,
            n.ANS: ANS,
            n.CAN: CAN,
            n.ANS_ID: ANS_ID,
        })

        total_loss += mean_loss
        total_accuracy += accuracy
        index += 1

        if index % 100 == 1:
            print('index: ', index,
                  'loss:', total_loss / index,
                  'accuracy:', accuracy,
                  'total accuracy:', total_accuracy / index)

    print('summary')
    print('loss:', total_loss / index)
    print('accuracy:', total_accuracy / index)
