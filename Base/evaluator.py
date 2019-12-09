import math
from collections import defaultdict

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
        self.ans_index = tf.argmax(tf.to_int32(tf.equal(rank, expand_ans)), axis=1) + 1

    def get_mrr(self):
        return tf.reduce_mean(1 / self.ans_index)

    def get_mean_rank(self):
        return tf.reduce_mean(self.ans_index)

    def get_top_x(self, index):
        return tf.reduce_mean(tf.to_float(self.ans_index <= index))


class SeqEvaluator:
    def __init__(self, preds: list, answers: list):
        for pred, ans in zip(preds, answers):
            bleu_stats = self.get_bleu_stats()

    @staticmethod
    def get_bleu_stats(ref, hyp, N=4):
        stats = defaultdict(int, {'rl': len(ref), 'hl': len(hyp)})
        N = len(hyp) if len(hyp) < N else N
        for n in range(N):
            matched = 0
            possible = defaultdict(int)
            for k in range(len(ref) - n):
                possible[tuple(ref[k: k + n + 1])] += 1
            for k in range(len(hyp) - n):
                ngram = tuple(hyp[k: k + n + 1])
                if possible[ngram] > 0:
                    possible[ngram] -= 1
                    matched += 1
            stats['d' + str(n + 1)] = len(hyp) - n
            stats['n' + str(n + 1)] = matched
        return stats

    @staticmethod
    def calculate_bleu(stats, N=4):
        np = 0.0
        for n in range(4):
            nn = stats['n' + str(n + 1)]
            if nn == 0:
                return 0.0
            dd = stats['d' + str(n + 1)]
            np += math.log(nn) - math.log(dd)
        bp = 1.0 - stats['rl'] / stats['hl']
        if bp > 0.0: bp = 0.0
        return math.exp(np / N + bp)
