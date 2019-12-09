import os

from nltk.translate.bleu_score import sentence_bleu
import tensorflow as tf

from Base.data_loader import DataLoader, END
from Config.qg_hp import qg_hp as hp
from net_qg import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

loader = DataLoader(hp, mode='test')

tf.reset_default_graph()

config = tf.ConfigProto(device_count={"GPU": 2})
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    n = Net(hp=hp)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(hp.logdir, 'epoch4.ckpt'))

    index = 0
    total_loss = 0
    total_accuracy = 0
    not_good_index = 0
    bleus = [0, 0, 0, 0]
    print('batch size:', hp.Data.batch_size)

    for VGG, C3D, HIS, LAST_QUE, ENID_QUE, LEN_QUE, RAW_QUE in loader.get_batch_data():
        loss, pred_question = sess.run(
            (n.loss, n.pred_question), feed_dict={
                n.VGG: VGG,
                n.C3D: C3D,
                n.HIS: HIS,
                n.ENID_QUE: ENID_QUE,
                n.LEN_QUE: LEN_QUE,
                n.LAST_QUE: LAST_QUE,
            })

        pred = pred_question.tolist()  # type: list
        accuracy = 0
        current_bleus = [0, 0, 0, 0]

        for b in range(hp.Data.batch_size):
            if END in pred[b]:
                pred[b] = pred[b][:pred[b].index(END)]
            bleu_candidate = [loader.idx2word[id_] for id_ in pred[b]]
            bleu_reference = RAW_QUE[b].split(' ')

            for bleu_index in range(4):
                weights = [0.0] * bleu_index + [1.0] + [0.0] * (3 - bleu_index)
                current_bleus[bleu_index] += sentence_bleu(
                    [bleu_reference], bleu_candidate, weights=weights)

            pred_str = ' '.join(bleu_candidate)
            if RAW_QUE[b] == pred_str:
                accuracy += 1
            else:
                if not_good_index % 50 == 0:
                    print('human', RAW_QUE[b])
                    print('model', pred_str)
                not_good_index += 1

        accuracy /= hp.Data.batch_size
        for bleu_index in range(4):
            current_bleus[bleu_index] /= hp.Data.batch_size
            bleus[bleu_index] += current_bleus[bleu_index]

        total_loss += loss
        total_accuracy += accuracy
        index += 1

        if index % 100 == 1:
            print('index: ', index,
                  'accuracy:', accuracy,
                  'loss:', total_loss / index,
                  'total accuracy:', total_accuracy / index,
                  'bleus', current_bleus)

    for bleu_index in range(4):
        bleus[bleu_index] /= index

    print('summary')
    print('loss:', total_loss / index)
    print('accuracy:', total_accuracy / index)
    print('bleus', bleus)

