import datetime
import os

from Base.crazy_data import CrazyData

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf

from Base.data_loader import DataLoader
from Base.modules import temporal_attention, label_smoothing
from Base.unit import mgsca_unit, sequence_embedding, gated_tangent
from Config.hyperparams import HyperParams


class Net:
    def __init__(self, hp: HyperParams = HyperParams, is_training=True):
        self.VGG = tf.placeholder(tf.float32, [
            hp.Data.batch_size, hp.Data.vgg_frames, hp.Data.vgg_feature_num], name='VGG')
        self.C3D = tf.placeholder(tf.float32, [
            hp.Data.batch_size, hp.Data.c3d_frames, hp.Data.c3d_feature_num], name='C3D')
        self.HIS = tf.placeholder(tf.float32, [
            hp.Data.batch_size,
            hp.Data.max_turn_per_dialog_len * 2,
            hp.Data.max_words_per_sentence_len,
            hp.Data.word_feature_num], name='HIS')
        self.QUE = tf.placeholder(tf.float32, [
            hp.Data.batch_size, hp.Data.max_words_per_sentence_len, hp.Data.word_feature_num],
                                  name='QUE')
        self.ANS = tf.placeholder(tf.float32, [
            hp.Data.batch_size, hp.Data.max_words_per_sentence_len, hp.Data.word_feature_num],
                                  name='ANS')
        self.CAN = tf.placeholder(tf.float32, [
            hp.Data.batch_size,
            hp.Data.candidate_num,
            hp.Data.max_words_per_sentence_len,
            hp.Data.word_feature_num], name='CAN')
        self.ANS_ID = tf.placeholder(tf.int32, [hp.Data.batch_size], name='ANS_ID')

        with tf.variable_scope('dialog_encoder'):
            turn_size = hp.Data.max_turn_per_dialog_len
            batch_size = hp.Data.batch_size
            # word_size = hp.Data.max_words_per_sentence_len
            # num_units = hp.Data.word_feature_num
            sentence_size = hp.Data.max_turn_per_dialog_len * 2

            history = tf.split(self.HIS, sentence_size, axis=1)
            # list of [batch_size, 1, max_words_per_sentence_len, word_feature_num]
            history = [tf.reshape(sentence, [
                hp.Data.batch_size, hp.Data.max_words_per_sentence_len, hp.Data.word_feature_num])
                       for sentence in history]
            # list of [batch_size, max_words_per_sentence_len, word_feature_num]

            history = [mgsca_unit(sentence,
                                  with_positional_encoding=False,
                                  scope='qa_MGCSA',
                                  is_training=is_training)
                       for sentence in history]

            history = [sequence_embedding(sentence) for sentence in history]
            # list of [batch_size, word_feature_num]

            with tf.variable_scope('joint_representation'):
                Wc1 = tf.Variable(tf.truncated_normal(
                    [hp.Data.word_feature_num, hp.Data.word_feature_num], stddev=0.1))
                Wc2 = tf.Variable(tf.truncated_normal(
                    [hp.Data.word_feature_num, hp.Data.word_feature_num], stddev=0.1))

                c = [tf.nn.tanh(
                    tf.matmul(history[index * 2], Wc1) + tf.matmul(history[index * 2 + 1], Wc2))
                     for index in range(turn_size)]  # list of [batch_size, word_feature_num]

            u = tf.reshape(
                tf.concat(c, axis=1), [
                    batch_size, turn_size, hp.Data.word_feature_num], name='reshapeC2U')
            u = mgsca_unit(u,
                           scope='masked_MGSCA_Unit',
                           using_mask=True,
                           is_training=is_training)
            # [batch_size, turn_size, word_feature_num]

            question = mgsca_unit(self.QUE,
                                  with_positional_encoding=False,
                                  scope='qa_MGCSA',
                                  is_training=is_training)
            question = sequence_embedding(question)  # [batch_size, word_feature_num]

            question = tf.add(question, temporal_attention(
                u, question, name='context_attention'))
            # [batch_size, word_feature_num]

        with tf.variable_scope('vgg_encoder'):
            vgg = mgsca_unit(self.VGG, is_training=is_training)
            vqf = temporal_attention(vgg, question, 'vgg_temporal_attention')
            # [batch_size, vgg_feature_num]

        with tf.variable_scope('c3d_encoder'):
            c3d = mgsca_unit(self.C3D, is_training=is_training)
            vqs = temporal_attention(c3d, question, 'c3d_temporal_attention')
            # [batch_size, c3d_feature_num]

        with tf.variable_scope('fusionX'):
            vq = tf.multiply(vqf, vqs)  # [batch_size, c3d_feature_num]
            fquv = tf.concat([
                gated_tangent(vq, middle_num_units=hp.middle_num_units, name='vq_gt'),
                gated_tangent(question, middle_num_units=hp.middle_num_units, name='q_gt')], axis=1)
            fquv = tf.layers.dense(fquv, hp.Data.word_feature_num, activation=tf.nn.relu)
            # [batch_size, word_feature_num]
            fquv = tf.expand_dims(fquv, axis=-1)
            # [batch_size, word_feature_num, 1]

        with tf.variable_scope('answer_decoder'):
            candidates = tf.unstack(self.CAN, num=hp.Data.candidate_num, axis=1)
            candidates = [mgsca_unit(candidate,
                                     scope='qa_MGCSA',
                                     is_training=is_training) for candidate in candidates]
            # list of [batch_size, word_feature_num]
            candidates = [sequence_embedding(candidate) for candidate in candidates]
            # list of [batch_size, word_feature_num]
            candidates = tf.reshape(tf.concat(candidates, axis=1), [
                batch_size, hp.Data.candidate_num, hp.Data.word_feature_num])

            logits = [tf.reshape(tf.matmul(
                candidates[index], fquv[index]), [hp.Data.candidate_num])
                for index in range(batch_size)]  # [batch_size, candidate_num]
            self.logits = tf.reshape(tf.concat(logits, axis=0), [batch_size, hp.Data.candidate_num])
            prediction = tf.to_int32(tf.argmax(logits, axis=-1))

        with tf.variable_scope('scoring'):
            ranks = tf.nn.top_k(logits, k=hp.Data.candidate_num)


        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(prediction, self.ANS_ID)))
        # tf.reduce_mean()

        if is_training:
            self.y_smoothed = label_smoothing(tf.one_hot(self.ANS_ID, depth=hp.Data.candidate_num))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss)

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta2=0.98)
            self.train_operation = self.optimizer.minimize(
                self.mean_loss, global_step=self.global_step)


if __name__ == '__main__':
    hp = HyperParams()
    loader = DataLoader(hp, mode='train')
    crazyData = CrazyData(
        pid='wzgE', ticket='iKdE5ycrTBzpESR7brvX4uJQN4kBdJsS9iSuTVWsygdAnM8YHTEEqQ3Fu9ugT9lp')

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        n = Net(hp=hp)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        for epoch in range(hp.epoch):
            index = 0
            print('epoch', epoch)
            for VGG, C3D, HIS, QUE, ANS, CAN, ANS_ID, turn_nums in loader.get_batch_data():
                _, mean_loss = sess.run((n.train_operation, n.mean_loss), feed_dict={
                    n.VGG: VGG,
                    n.C3D: C3D,
                    n.HIS: HIS,
                    n.QUE: QUE,
                    n.ANS: ANS,
                    n.CAN: CAN,
                    n.ANS_ID: ANS_ID,
                })
                if index % 100 == 0:
                    print(str(datetime.datetime.now()), 'index', index, 'loss', mean_loss)
                    crazyData.push([dict(label='loss', value=int(mean_loss*1000))])
                index += 1
            print('total index', index, 'start saving!')
            saver.save(sess, os.path.join(hp.logdir, 'epoch{0}.ckpt'.format(epoch)))
