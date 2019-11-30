import tensorflow as tf


class Evaluator:
    def __init__(self, logits: tf.Tensor, ans: tf.Tensor):
        """
        :param logits: [batch_size, candidate_num] float32
        :param ans: [batch_size] int32
        """
        batch_size, candidate_num = logits.get_shape().as_list()
        _, self.rank = tf.nn.top_k(logits, k=candidate_num)
        self.ans = ans

    def get_mrr(self):
        