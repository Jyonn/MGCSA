import datetime
import json
import os

import numpy as np
import tensorflow as tf

from Base.crazy_data import CrazyData
from Base.data_loader import DataLoader, END
from Config.qg_hp import qg_hp as hp
from net_qg import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

loader = DataLoader(hp, mode='train')
# crazyData = CrazyData(
#     pid='lUGI', ticket='80Hv3zUI6v3jbwj4o3SgZZ7evkXedZE8FxI472GuLoeoiqwCeFjZHwBexBYgxos5')

tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    n = Net(hp=hp)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for epoch in range(hp.epoch):
        index = 0
        output = None
        print('epoch', epoch)
        for VGG, C3D, HIS, QUE, ENID_QUE, LEN_QUE, RAW_QUE, ANS, CAN, ANS_ID in loader.get_batch_data():
            last_output = output  # type: np.ndarray
            _, loss, pred_question, output = sess.run(
                (n.train_operation, n.loss, n.pred_question, n.lstm_output), feed_dict={
            # masks, max_question_length = sess.run(
            #     (n.masks, n.max_question_length), feed_dict={
                    n.VGG: VGG,
                    n.C3D: C3D,
                    n.HIS: HIS,
                    n.ENID_QUE: ENID_QUE,
                    n.LEN_QUE: LEN_QUE
                })

            # print(masks)
            # exit(0)

            if index % 100 == 0:
                pred = pred_question.tolist()[0]  # type: list
                if END in pred:
                    pred = pred[:pred.index(END)]
                print('human:', RAW_QUE[0])
                print('model:', ' '.join([loader.idx2word[id_] for id_ in pred]))
                if index:
                    last = json.dumps(last_output.tolist())
                    current = json.dumps(output.tolist())
                    with open('last-output.txt', 'wb+') as f:
                        f.write(last.encode())
                    with open('current-output.txt', 'wb+') as f:
                        f.write(current.encode())
                    print('save output at', index)
                    # print('same?:', (last_output == output).all())
                print(str(datetime.datetime.now()),
                      'index:', index,
                      'loss:', loss)
                # crazyData.push([
                #     dict(label='loss', value=int(loss * 1000))])
            index += 1
        print('total index', index, 'start saving!')
        saver.save(sess, os.path.join(hp.logdir, 'epoch{0}.ckpt'.format(epoch)))
