import tensorflow as tf


class Evaluator:
    def __init__(self, logits: tf.Tensor, ans: tf.Tensor):
        """
        :param logits: [batch_size, candidate_num] float32
        :param ans: [batch_size] int32
        """
        batch_size, candidate_num = logits.get_shape().as_list()
        _, rank = tf.nn.top_k(logits, k=candidate_num)  # [batch_size, candidate_num]
        expand_ans = tf.tile(tf.expand_dims(ans, 1), [1, candidate_num])
        self.ans_index = tf.argmax(tf.to_int32(tf.equal(rank, expand_ans))) + 1

    def get_mrr(self):
        return tf.reduce_mean(1 / self.ans_index)

    def get_mean_rank(self):
        return tf.reduce_mean(self.ans_index)

    def get_top_x(self, index):
        return tf.reduce_mean(tf.to_float(self.ans_index <= index))
