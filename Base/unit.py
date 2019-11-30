import tensorflow as tf

from Base.modules import positional_encoding, scaled_dotproduct_attention, feedforward, fusion
from Config.hyperparams import HyperParams as hp


def mgsca_unit(inputs: tf.Tensor,
               scope='MGSCA_Unit',
               reuse=None,
               is_training=True,
               with_positional_encoding=True,
               using_mask=False):
    """
    MGSCA Unit

    :param num_units: num_units
    :param with_positional_encoding: using positional encoding
    :param is_training: Boolean. Controller of mechanism for dropout.
    :param inputs: [batch_size, feature_num, embedding_size]. Feature_num is "sentence_length"
    when inputs are dialogues; it's "frame_length" when inputs are video frames. It has been
    position-embedded.
    :param scope: Optional scope.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :param using_mask: Boolean. If true, units that reference the future are masked.

    :return:
    """

    with tf.variable_scope(scope, reuse=reuse):
        batch_size, feature_num, embedding_length = inputs.get_shape().as_list()
        encoded = inputs

        if with_positional_encoding:
            encoded = tf.add(inputs, positional_encoding([batch_size, feature_num],
                                                         num_units=embedding_length,
                                                         zero_pad=False,
                                                         scale=False,
                                                         scope='positional_encoding'))
            encoded = tf.layers.dropout(encoded, rate=hp.dropout_rate,
                                        training=tf.convert_to_tensor(is_training))

        assert feature_num % hp.kernel_size == 0
        encoded = tf.reshape(encoded, [-1, hp.kernel_size, embedding_length])

        y = scaled_dotproduct_attention(queries=encoded,
                                        keys=encoded,
                                        dropout_rate=hp.dropout_rate,
                                        is_training=is_training,
                                        using_mask=using_mask,
                                        reuse=tf.AUTO_REUSE)

        y = tf.reshape(y, [-1, feature_num, embedding_length])
        # [batch_size, feature_num, hp.num_units]

        p = tf.layers.conv1d(y,
                             filters=embedding_length,
                             kernel_size=hp.kernel_size,
                             strides=hp.kernel_size,
                             reuse=tf.AUTO_REUSE)
        # [batch_size, kernel_num, embedding_length]

        p2 = scaled_dotproduct_attention(queries=p,
                                         keys=p,
                                         dropout_rate=hp.dropout_rate,
                                         is_training=is_training,
                                         using_mask=False,
                                         scope='att_forwarded',
                                         reuse=tf.AUTO_REUSE)
        # [batch_size, kernel_num, embedding_length]

        z2 = fusion(p, p2, reuse=tf.AUTO_REUSE)
        z = tf.tile(z2, [1, hp.kernel_size, 1])
        fyz = fusion(y, z, reuse=tf.AUTO_REUSE)
        r = fusion(fyz, inputs, reuse=tf.AUTO_REUSE)
        # fusioned = tf.layers.dense(fusioned, num_units or hp.num_units, reuse=tf.AUTO_REUSE)
        return r  # [batch_size, feature_num, embedding_length]


# def get_dialog_size(DLG: tf.Tensor):
#     dialog_sum = tf.reduce_sum(tf.abs(DLG), axis=-1)
#     dialog_sum = tf.reduce_sum(dialog_sum, axis=-1)
#     return tf.div(tf.count_nonzero(dialog_sum, axis=1), 2)


def sequence_embedding(
        inputs: tf.Tensor,
        middle_num_units=None,
        name='sequence_embedding'):
    num_units = inputs.get_shape().as_list()[-1]
    middle_num_units = middle_num_units or num_units
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, middle_num_units, activation=tf.nn.tanh, name='seD1')
        outputs = tf.layers.dense(outputs, num_units, activation=tf.nn.softmax, name='seD2')
        multiply = tf.multiply(outputs, inputs)
        outputs = tf.reduce_sum(multiply, axis=1)
    return outputs  # [batch_size, num_units]


def gated_tangent(
        inputs: tf.Tensor,
        middle_num_units=None,
        name='gated_tangent_activation'):
    num_units = inputs.get_shape().as_list()[-1]
    middle_num_units = middle_num_units or num_units
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, middle_num_units, activation=tf.nn.tanh, name='gtD1')
        outputs = tf.layers.dense(outputs, num_units, activation=tf.nn.softmax, name='gtD2')
        outputs = tf.multiply(outputs, inputs)
    return outputs  # [batch_size, num_units]
