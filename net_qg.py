from tensorflow.contrib.seq2seq import sequence_loss

from Base.evaluator import Evaluator

import tensorflow as tf

from Base.modules import temporal_attention, label_smoothing
from Base.unit import mgsca_unit, sequence_embedding, gated_tangent
from Config.hyperparams import HyperParams


class Net:
    class RnnLoop:
        def __init__(self, initial_state, cell, hp: HyperParams):
            self.initial_state = initial_state
            self.cell = cell
            self.hp = hp

        def __call__(self, time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                initial_input = tf.fill([self.hp.Data.batch_size, self.hp.Data.c3d_feature_num], 0.0)
                next_input = initial_input
                next_cell_state = self.initial_state
            else:
                next_input = cell_output
                next_cell_state = cell_state

            elements_finished = (time >= self.hp.Data.max_words_per_sentence_len)
            next_loop_state = None
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

    def utterance_encoder(self, hp: HyperParams, is_training=True):
        with tf.variable_scope('utterance_encoder'):
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
            return u

    def __init__(self, hp: HyperParams, is_training=True):
        self.VGG = tf.placeholder(tf.float32, [
            hp.Data.batch_size, hp.Data.vgg_frames, hp.Data.vgg_feature_num], name='VGG')
        self.C3D = tf.placeholder(tf.float32, [
            hp.Data.batch_size, hp.Data.c3d_frames, hp.Data.c3d_feature_num], name='C3D')
        self.HIS = tf.placeholder(tf.float32, [
            hp.Data.batch_size,
            hp.Data.max_turn_per_dialog_len * 2,
            hp.Data.max_words_per_sentence_len,
            hp.Data.word_feature_num], name='HIS')
        # self.QUE = tf.placeholder(tf.float32, [
        #     hp.Data.batch_size, hp.Data.max_words_per_sentence_len, hp.Data.word_feature_num],
        #                           name='QUE')
        self.ENID_QUE = tf.placeholder(tf.int32, [
            hp.Data.batch_size, hp.Data.max_words_per_sentence_len], name='ENID_QUE')
        self.LEN_QUE = tf.placeholder(tf.int32, [
            hp.Data.batch_size], name='LEN_QUE')
        # self.ANS = tf.placeholder(tf.float32, [
        #     hp.Data.batch_size, hp.Data.max_words_per_sentence_len, hp.Data.word_feature_num],
        #                           name='ANS')
        # self.CAN = tf.placeholder(tf.float32, [
        #     hp.Data.batch_size,
        #     hp.Data.candidate_num,
        #     hp.Data.max_words_per_sentence_len,
        #     hp.Data.word_feature_num], name='CAN')
        # self.ANS_ID = tf.placeholder(tf.int32, [hp.Data.batch_size], name='ANS_ID')

        with tf.variable_scope('question_generator'):
            batch_size = hp.Data.batch_size
            u = self.utterance_encoder(hp=hp, is_training=is_training)
            # [batch_size, turn_size, word_feature_num]
            vgg = mgsca_unit(self.VGG, is_training=is_training)
            # [batch_size, vgg_frames, vgg_feature_num]
            c3d = mgsca_unit(self.C3D, is_training=is_training)
            # [batch_size, c3d_frames, c3d_feature_num]

            u_qg = sequence_embedding(u, name='utterance_se')  # [batch_size, word_feature_num]
            vqf_qg = temporal_attention(vgg, u_qg, 'vgg_qg_ta')
            vqs_qg = temporal_attention(c3d, u_qg, 'c3d_qg_ta')
            vq_qg = tf.multiply(vqf_qg, vqs_qg)  # [batch_size, c3d_feature_num]

            # self.lstm_init_state = tf.layers.dense(
            #     vq_qg, hp.middle_num_units, activation=tf.nn.relu)  # [batch_size, middle_num_units]
            # lstm_cell = tf.nn.rnn_cell.GRUCell(hp.middle_num_units)
            self.lstm_init_state = vq_qg
            lstm_cell = tf.nn.rnn_cell.GRUCell(hp.Data.c3d_feature_num)
            rnn_loop = self.RnnLoop(initial_state=self.lstm_init_state, cell=lstm_cell, hp=hp)
            self.lstm_output, _, _ = tf.nn.raw_rnn(lstm_cell, rnn_loop)  # type: tf.TensorArray, tf.Tensor, tf.Tensor
            self.lstm_output = self.lstm_output.stack()
            self.lstm_output = tf.transpose(self.lstm_output, [1, 0, 2])
            # [batch_size, max_words_per_sentence_len, middle_num_units]

            # self.lstm_output = tf.reshape(self.lstm_output, [
            #     batch_size * hp.Data.max_words_per_sentence_len, hp.middle_num_units])
            self.lstm_output = tf.layers.dense(self.lstm_output,
                                               hp.Data.vocab_size,
                                               activation=tf.nn.softmax,
                                               kernel_initializer=tf.random_normal_initializer())
            self.lstm_output = tf.reshape(self.lstm_output, [
                batch_size, hp.Data.max_words_per_sentence_len, hp.Data.vocab_size])

            self.pred_question = tf.argmax(self.lstm_output, axis=-1)
            # [batch_size, max_words_per_sentence_len]

            if is_training:
                # self.max_question_length = tf.reduce_max(self.LEN_QUE, name='max_question_len')
                self.masks = tf.sequence_mask(
                    self.LEN_QUE, hp.Data.max_words_per_sentence_len, dtype=tf.float32)

                # print('len:', self.LEN_QUE.shape)
                # print('mask:', self.masks.shape)
                # print('id:', self.ENID_QUE.shape)
                # print('output:', self.lstm_output.shape)
                self.loss = sequence_loss(self.lstm_output, self.ENID_QUE, weights=self.masks)

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.learning_rate, beta2=0.98)
                self.train_operation = self.optimizer.minimize(
                    self.loss, global_step=self.global_step)

        #     question = mgsca_unit(self.QUE,
        #                           with_positional_encoding=False,
        #                           scope='qa_MGCSA',
        #                           is_training=is_training)
        #     question = sequence_embedding(question)  # [batch_size, word_feature_num]
        #
        #     question = tf.add(question, temporal_attention(
        #         u, question, name='context_attention'))
        #     # [batch_size, word_feature_num]
        #
        # with tf.variable_scope('vgg_encoder'):
        #     vqf = temporal_attention(vgg, question, 'vgg_temporal_attention')
        #     # [batch_size, vgg_feature_num]
        #
        # with tf.variable_scope('c3d_encoder'):
        #     vqs = temporal_attention(c3d, question, 'c3d_temporal_attention')
        #     # [batch_size, c3d_feature_num]
        #
        # with tf.variable_scope('fusionX'):
        #     vq = tf.multiply(vqf, vqs)  # [batch_size, c3d_feature_num]
        #     fquv = tf.concat([
        #         gated_tangent(vq, middle_num_units=hp.middle_num_units, name='vq_gt'),
        #         gated_tangent(question, middle_num_units=hp.middle_num_units, name='q_gt')], axis=1)
        #     fquv = tf.layers.dense(fquv, hp.Data.word_feature_num, activation=tf.nn.relu)
        #     # [batch_size, word_feature_num]
        #     fquv = tf.expand_dims(fquv, axis=-1)
        #     # [batch_size, word_feature_num, 1]
        #
        # with tf.variable_scope('answer_decoder'):
        #     candidates = tf.unstack(self.CAN, num=hp.Data.candidate_num, axis=1)
        #     candidates = [mgsca_unit(candidate,
        #                              scope='qa_MGCSA',
        #                              is_training=is_training) for candidate in candidates]
        #     # list of [batch_size, word_feature_num]
        #     candidates = [sequence_embedding(candidate) for candidate in candidates]
        #     # list of [batch_size, word_feature_num]
        #     candidates = tf.reshape(tf.concat(candidates, axis=1), [
        #         batch_size, hp.Data.candidate_num, hp.Data.word_feature_num])
        #
        #     logits = [tf.reshape(tf.matmul(
        #         candidates[index], fquv[index]), [hp.Data.candidate_num])
        #         for index in range(batch_size)]  # [batch_size, candidate_num]
        #     self.logits = tf.reshape(tf.concat(logits, axis=0), [batch_size, hp.Data.candidate_num])
        #
        # with tf.variable_scope('scoring'):
        #     self.evaluator = Evaluator(self.logits, self.ANS_ID)
        #     self.accuracy = self.evaluator.get_top_x(1)
        #     self.pAt1 = self.accuracy
        #     self.pAt5 = self.evaluator.get_top_x(5)
        #     self.mrr = self.evaluator.get_mrr()
        #     self.meanRank = self.evaluator.get_mean_rank()
        #
        # if is_training:
        #     self.y_smoothed = label_smoothing(tf.one_hot(self.ANS_ID, depth=hp.Data.candidate_num))
        #     self.loss = tf.nn.softmax_cross_entropy_with_logits(
        #         logits=logits, labels=self.y_smoothed)
        #     self.mean_loss = tf.reduce_sum(self.loss)
        #
        #     self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.learning_rate, beta2=0.98)
        #     self.train_operation = self.optimizer.minimize(
        #         self.mean_loss, global_step=self.global_step)


if __name__ == '__main__':
    Net(HyperParams())
