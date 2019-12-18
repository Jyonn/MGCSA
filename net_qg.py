from tensorflow.contrib.seq2seq import sequence_loss


import tensorflow as tf

from Base.modules import temporal_attention
from Base.unit import mgsca_unit, sequence_embedding, gated_tangent
from Config.hyperparams import HyperParams


class Net:
    @staticmethod
    def get_rnn_cell(units):
        def get_single_rnn_cell(num_units):
            gru_cell = tf.nn.rnn_cell.GRUCell(num_units)
            return tf.nn.rnn_cell.DropoutWrapper(gru_cell)

        return tf.nn.rnn_cell.MultiRNNCell([get_single_rnn_cell(units) for _ in range(2)])

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
                                  is_training=is_training,
                                  kernel_size=hp.sentence_kernel_size)
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
                           is_training=is_training,
                           kernel_size=hp.sentence_kernel_size)
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
        self.LAST_QUE = tf.placeholder(tf.float32, [
            hp.Data.batch_size, hp.Data.max_words_per_sentence_len, hp.Data.word_feature_num],
                                  name='LAST_QUE')
        # self.QUE = tf.placeholder(tf.float32, [
        #     hp.Data.batch_size, hp.Data.max_words_per_sentence_len, hp.Data.word_feature_num],
        #                           name='QUE')
        self.ENID_QUE = tf.placeholder(tf.int32, [
            hp.Data.batch_size, hp.Data.max_words_per_sentence_len], name='ENID_QUE')
        self.LEN_QUE = tf.placeholder(tf.int32, [
            hp.Data.batch_size], name='LEN_QUE')

        with tf.variable_scope('dialog_encoder'):
            u = self.utterance_encoder(hp, is_training)
            question = mgsca_unit(self.LAST_QUE,
                                  with_positional_encoding=False,
                                  scope='qa_MGCSA',
                                  is_training=is_training,
                                  kernel_size=hp.sentence_kernel_size)
            question = sequence_embedding(question)  # [batch_size, word_feature_num]

            question = tf.add(question, temporal_attention(
                u, question, name='context_attention'))
            # [batch_size, word_feature_num]

        with tf.variable_scope('vgg_encoder'):
            vgg = mgsca_unit(self.VGG, is_training=is_training, kernel_size=hp.video_kernel_size)
            vqf = temporal_attention(vgg, question, 'vgg_temporal_attention')
            # [batch_size, vgg_feature_num]

        with tf.variable_scope('c3d_encoder'):
            c3d = mgsca_unit(self.C3D, is_training=is_training, kernel_size=hp.video_kernel_size)
            vqs = temporal_attention(c3d, question, 'c3d_temporal_attention')
            # [batch_size, c3d_feature_num]

        with tf.variable_scope('fusionX'):
            vq = tf.multiply(vqf, vqs)  # [batch_size, c3d_feature_num]
            fquv = tf.concat([
                gated_tangent(vq, middle_num_units=hp.middle_num_units, name='vq_gt'),
                gated_tangent(question, middle_num_units=hp.middle_num_units, name='q_gt')], axis=1)
            fquv = tf.layers.dense(fquv, hp.Data.word_feature_num, activation=tf.nn.relu)
            # [batch_size, word_feature_num]

        with tf.variable_scope('question_generator'):
            batch_size = hp.Data.batch_size

            # rnn_cell = self.get_rnn_cell(hp.Data.word_feature_num)

            rnn_cell = tf.nn.rnn_cell.GRUCell(hp.Data.word_feature_num)
            self.rnn_output, _ = tf.nn.dynamic_rnn(rnn_cell, self.LAST_QUE, initial_state=fquv)

            # # tf.nn.dynamic_rnn(rnn_cell, )
            # rnn_loop = self.RnnLoop(initial_state=self.lstm_init_state, cell=rnn_cell, hp=hp)
            # self.rnn_output, _, _ = tf.nn.raw_rnn(rnn_cell, rnn_loop)  # type: tf.TensorArray, tf.Tensor, tf.Tensor
            # self.rnn_output = self.rnn_output.stack()
            # shape: [batch_size, max_words_per_sentence_len, middle_num_units]

            self.rnn_output = tf.layers.dense(self.rnn_output,
                                              hp.Data.vocab_size,
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer())
            self.rnn_output = tf.reshape(self.rnn_output, [
                batch_size, hp.Data.max_words_per_sentence_len, hp.Data.vocab_size])

            self.pred_question = tf.argmax(self.rnn_output, axis=-1)
            # [batch_size, max_words_per_sentence_len]

            if is_training:
                # self.max_question_length = tf.reduce_max(self.LEN_QUE, name='max_question_len')
                self.masks = tf.sequence_mask(
                    self.LEN_QUE, hp.Data.max_words_per_sentence_len, dtype=tf.float32)
                self.loss = sequence_loss(self.rnn_output, self.ENID_QUE, weights=self.masks)

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.learning_rate)
                self.train_operation = self.optimizer.minimize(
                    self.loss, global_step=self.global_step)


if __name__ == '__main__':
    Net(HyperParams())
