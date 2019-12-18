import datetime
import json
import os

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu

from Base.crazy_data import CrazyData
from Base.data_loader import DataLoader, END
from Config.qg_hp import qg_hp_TACoS as hp
from net_qg import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

loader = DataLoader(hp, mode='train')
crazyData = CrazyData(
    pid='IO4c', ticket='WETG8N4Y2HCQjgbSzHnU9JDTLs6jpLPX2I5G34HfMlbJQDQHJfa4QsPMX72zG5UD')

tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    n = Net(hp=hp)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(hp.logdir, 'epoch3.ckpt'))

    for epoch in range(4, hp.epoch):
        index = 0
        output = None
        # bleus = [0, 0, 0, 0]
        print('epoch', epoch)
        for VGG, C3D, HIS, LAST_QUE, ENID_QUE, LEN_QUE, RAW_QUE in loader.get_batch_data():
            last_output = output  # type: np.ndarray
            _, loss, pred_question, output = sess.run(
                (n.train_operation, n.loss, n.pred_question, n.rnn_output), feed_dict={
            # masks, max_question_length = sess.run(
            #     (n.masks, n.max_question_length), feed_dict={
                    n.VGG: VGG,
                    n.C3D: C3D,
                    n.HIS: HIS,
                    n.ENID_QUE: ENID_QUE,
                    n.LEN_QUE: LEN_QUE,
                    n.LAST_QUE: LAST_QUE,
                })

            # print(masks)
            # exit(0)

            if index % 100 == 0:
                current_bleus = [0, 0, 0, 0]
                pred = pred_question.tolist()

                print_it = True
                for b in range(hp.Data.batch_size):
                    if END in pred[b]:
                        pred[b] = pred[b][:pred[b].index(END)]
                    bleu_candidate = [loader.idx2word[id_] for id_ in pred[b]]
                    bleu_reference = RAW_QUE[b].split(' ')

                    for bleu_index in range(4):
                        weights = [1 / (bleu_index + 1)] * (bleu_index + 1) + [0.0] * (
                                    3 - bleu_index)
                        current_bleus[bleu_index] += sentence_bleu(
                            [bleu_reference], bleu_candidate, weights=weights)

                    pred_str = ' '.join(bleu_candidate)

                    if print_it:
                        print('human:', RAW_QUE[b])
                        print('model:', pred_str)
                        print_it = False

                for bleu_index in range(4):
                    current_bleus[bleu_index] /= hp.Data.batch_size

                print(str(datetime.datetime.now()),
                      'index:', index,
                      'loss:', loss,
                      'bleu:', current_bleus)

                crazyData.push([
                    dict(label='loss@e%s' % epoch, value=int(loss * 1000)),
                    dict(label='belu1@e%s' % epoch, value=int(current_bleus[0] * 10000)),
                    dict(label='belu2@e%s' % epoch, value=int(current_bleus[1] * 10000))
                ])
            index += 1
        print('total index', index, 'start saving!')
        saver.save(sess, os.path.join(hp.logdir, 'epoch{0}.ckpt'.format(epoch)))
