import json
import random

import redis
import pickle as pkl
import numpy as np

from Config.hyperparams import HyperParams
from Data.TACoS.train_dialog_turns import len_matrix as TACoS_train_len_matrix
from Data.TACoS.test_dialog_turns import len_matrix as TACoS_test_len_matrix
from Data.YoutubeClip.train_dialog_turns import len_matrix as YoutubeClip_train_len_matrix
from Data.YoutubeClip.test_dialog_turns import len_matrix as YoutubeClip_test_len_matrix

PAD = 0
UNK = 1
BEG = 2
END = 3


class DataLoader:
    def __init__(self, hp: HyperParams, mode='train'):
        self.hp = hp
        self.mode = mode

        if hp.dataset == 'TACoS':
            self.len_matrix = TACoS_train_len_matrix if mode == 'train' else TACoS_test_len_matrix
        else:
            self.len_matrix = YoutubeClip_train_len_matrix if mode == 'train' else YoutubeClip_test_len_matrix

        self.word2idx, self.idx2word, self.word_len = self.load_words()
        self.word_embedding = self.load_word_embedding()

        self.rdb = redis.StrictRedis(hp.Data.redis_host, port=hp.Data.redis_port)
        self.num_records = hp.Data.train_records if mode == 'train' else hp.Data.test_records

    def load_words(self):
        with open(self.hp.Data.word_file, 'r') as f:
            words = f.readlines()
            words = list(map(lambda x: x[:-1], words))
            word2idx = {word: idx for idx, word in enumerate(words)}
            idx2word = {idx: word for idx, word in enumerate(words)}
            return word2idx, idx2word, len(words)

    def load_word_embedding(self):
        matrix = pkl.load(open(self.hp.Data.word_embedding, 'rb'), encoding='latin1')
        symbols = np.zeros((4, matrix.shape[-1]), np.float32)
        return np.vstack((symbols, matrix))

    def encode_sentence(self, sentence: str, max_len):
        if not sentence:
            return [PAD] * max_len
        words = sentence.split(' ')
        words = list(map(lambda w: self.word2idx.get(w) or UNK, words))
        # words.insert(0, BEG)
        words = words[:max_len - 1]
        words.append(END)
        words.extend([PAD] * (max_len - len(words)))
        return words

    def get_answer_info(self, answer_id):
        return pkl.loads(self.rdb.get('a{0}'.format(answer_id)), encoding='latin1')

    def load_answer(self, turn):
        answer_info = self.get_answer_info(turn[1])

        candidate_ids = answer_info[1][:self.hp.Data.candidate_num]
        if not candidate_ids.__contains__(turn[1]):
            candidate_ids[0] = turn[1]

        candidates = [self.get_answer_info(candidate_id)[0] for candidate_id in candidate_ids]
        candidates.extend([None] * (self.hp.Data.candidate_num - len(candidates)))
        random.shuffle(candidates)
        answer = answer_info[0]
        return turn[0], answer, candidates, candidates.index(answer)

    def load_record(self, record_id):

        def enid(turn):
            return list(map(
                lambda s: self.encode_sentence(s, self.hp.Data.max_words_per_sentence_len), turn))

        def embedding(turn):
            sentences = list(map(
                lambda s: self.encode_sentence(s, self.hp.Data.max_words_per_sentence_len), turn))
            # return [list(map(lambda i: self.word_embedding[i], sentence))
            # for sentence in sentences]
            return [self.word_embedding[sentence] for sentence in sentences]

        def get_length(turn):
            return list(map(lambda s: len(s.split(' ')) + 1, turn))

        record = self.rdb.get(self.mode + str(record_id))
        record = pkl.loads(record)

        dialog = list(map(self.load_answer, record['dialog']))
        enid_dialog = [enid(_turn[:2]) for _turn in dialog]
        len_dialog = [get_length(_turn[:2]) for _turn in dialog]
        raw_que = [_turn[0] for _turn in dialog]
        embedded_dialog = [embedding(_turn[:2]) for _turn in dialog]
        embedded_dialog = np.vstack(embedded_dialog)
        dialog_candidate = [embedding(_turn[2]) for _turn in dialog]
        dialog_answer = [_turn[3] for _turn in dialog]
        embedded_candidate = np.vstack(dialog_candidate)

        video_feature = pkl.loads(self.rdb.get(record['video_feature']), encoding='latin1')
        return (video_feature['vgg'], video_feature['c3d'],
                embedded_dialog, embedded_candidate, dialog_answer, enid_dialog, len_dialog, raw_que)

    def load_readable_record(self, record_id):
        record = self.rdb.get(self.mode + str(record_id))
        record = pkl.loads(record)

        dialog = list(map(self.load_answer, record['dialog']))
        candidate = [_turn[2] for _turn in dialog]
        answer = [_turn[3] for _turn in dialog]
        dialog = [_turn[:2] for _turn in dialog]
        question = [_turn[0] for _turn in dialog]
        return dialog, candidate, answer, question

    def get_dialog_turns(self):
        def load_answer(turn):
            answer = pkl.loads(self.rdb.get('a{0}'.format(turn[1])), encoding='latin1')[0]
            return turn[0], answer

        lens = []
        for index in range(self.num_records):
            record = self.rdb.get(self.mode + str(index))
            record = pkl.loads(record)
            dialog = list(map(load_answer, record['dialog']))
            lens.append(len(dialog))

        len_matrix = [[]] * self.hp.Data.max_turn_per_dialog_len
        for index, l in enumerate(lens):
            if l >= self.hp.Data.max_turn_per_dialog_len:
                print('too long dialog', index, l)
                continue
            if len_matrix[l]:
                len_matrix[l].append(index)
            else:
                len_matrix[l] = [index]

        with open('./Data/{1}/{0}_dialog_turns.py'.format(self.mode, self.hp.dataset), 'wb+') as f:
            s = json.dumps(len_matrix)
            s = s.replace('null', 'None')
            s = 'len_matrix = {0}'.format(s)
            s = s.encode()
            f.write(s)

    def get_total_batches(self, keep_remainder=True, length_above=1):
        total_batches = 0
        for length in range(length_above, self.hp.Data.max_turn_per_dialog_len):
            current_batches = len(self.len_matrix[length]) // self.hp.Data.batch_size
            if keep_remainder and len(self.len_matrix[length]) % self.hp.Data.batch_size:
                current_batches += 1
            current_batches *= length - 1
            total_batches += current_batches
        return total_batches

    def get_record_ids(self, keep_remainder=True, length_above=1):
        batch_size = self.hp.Data.batch_size
        print_it = True
        # if self.mode == 'train':
        turn_range = range(length_above, self.hp.Data.max_turn_per_dialog_len)
        # else:
        #     turn_range = range(length_above, length_above+1)
        for turn_num in turn_range:
            lens = len(self.len_matrix[turn_num])
            if print_it:
                print('randomize... [:10]', self.len_matrix[turn_num][:10])
            random.shuffle(self.len_matrix[turn_num])
            if print_it:
                print('randomization finished. ', self.len_matrix[turn_num][:10])
                print_it = False
            batch_num = lens // batch_size
            if keep_remainder and lens % batch_size:
                batch_num += 1
            for batch_index in range(batch_num):
                index_start = batch_size * batch_index
                index_end = min(batch_size * (batch_index + 1), lens)
                yield turn_num, self.len_matrix[turn_num][index_start:index_end]

    def get_readable_batch(self):
        for turn_nums, record_ids in self.get_record_ids(length_above=2, keep_remainder=False):
            dlg_list = []
            can_list = []
            ans_list = []
            que_list = []

            current_batch_size = 0
            for record_id in record_ids:
                current_batch_size += 1
                dialog, candidate, answer, question = self.load_readable_record(record_id)
                dialog_ = []
                for turn in dialog:
                    dialog_.append(turn[0])
                    dialog_.append(turn[1])
                dlg_list.append(dialog_)
                can_list.append(candidate)
                ans_list.append(answer)
                que_list.append(question)

            DLG = []
            for sentence_num in range(turn_nums * 2):
                TURN_DLG = []
                for batch_num in range(current_batch_size):
                    TURN_DLG.append(dlg_list[batch_num][sentence_num])
                DLG.append(TURN_DLG)

            CAN = []
            ANS = []
            QUE = []
            for turn_num in range(turn_nums):
                TURN_CAN = []
                TURN_ANS = []
                TURN_QUE = []
                for batch_num in range(current_batch_size):
                    TURN_CAN.append(can_list[batch_num][turn_num])
                    TURN_ANS.append(ans_list[batch_num][turn_num])
                    TURN_QUE.append(que_list[batch_num][turn_num])
                CAN.append(TURN_CAN)
                ANS.append(TURN_ANS)
                QUE.append(TURN_QUE)

            for turn_num in range(1, turn_nums):
                CRT_DLG = DLG[:turn_num * 2]
                LAST_QUE = DLG[turn_num * 2 - 2]
                QUE = DLG[turn_num * 2]

                yield CRT_DLG, LAST_QUE, QUE

    def get_batch_data(self):
        for turn_nums, record_ids in self.get_record_ids(length_above=2, keep_remainder=False):
            vgg_list = []
            c3d_list = []
            dlg_list = []
            can_list = []
            ans_list = []
            enid_list = []
            len_list = []
            raw_list = []

            current_batch_size = 0
            for record_id in record_ids:
                current_batch_size += 1
                vgg, c3d, dialog, candidate, answer, enid_dlg, len_dlg, raw_que = self.load_record(record_id)
                vgg_list.append(vgg)
                c3d_list.append(c3d)
                dlg_list.append(dialog)
                can_list.append(candidate)
                ans_list.append(answer)
                enid_list.append(enid_dlg)
                len_list.append(len_dlg)
                raw_list.append(raw_que)

            VGG = np.array(vgg_list, dtype=np.float32).reshape([
                current_batch_size, self.hp.Data.vgg_frames, self.hp.Data.vgg_feature_num])
            C3D = np.array(c3d_list, dtype=np.float32).reshape([
                current_batch_size, self.hp.Data.c3d_frames, self.hp.Data.c3d_feature_num])
            DLG = np.array(dlg_list, dtype=np.float32).reshape([
                current_batch_size,
                turn_nums * 2,
                self.hp.Data.max_words_per_sentence_len,
                self.hp.Data.word_feature_num])
            DLG = np.split(DLG, turn_nums * 2, axis=1)

            ENID_DLG = np.array(enid_list, dtype=np.int32).reshape([
                current_batch_size,
                turn_nums * 2,
                self.hp.Data.max_words_per_sentence_len])
            ENID_DLG = np.split(ENID_DLG, turn_nums * 2, axis=1)
            LEN_DLG = np.array(len_list, dtype=np.int32).reshape([
                current_batch_size, turn_nums * 2])
            LEN_DLG = np.split(LEN_DLG, turn_nums * 2, axis=1)

            CAN = np.array(can_list, dtype=np.float32).reshape([
                current_batch_size,
                turn_nums,
                self.hp.Data.candidate_num,
                self.hp.Data.max_words_per_sentence_len,
                self.hp.Data.word_feature_num])
            CAN = np.split(CAN, turn_nums, axis=1)
            CAN = [np.reshape(x, [
                current_batch_size,
                self.hp.Data.candidate_num,
                self.hp.Data.max_words_per_sentence_len,
                self.hp.Data.word_feature_num]) for x in CAN]

            ANS_ID = np.array(ans_list, dtype=np.int32).reshape([
                current_batch_size, turn_nums])
            ANS_ID = np.split(ANS_ID, turn_nums, axis=1)
            ANS_ID = [np.reshape(x, [current_batch_size]) for x in ANS_ID]

            empty_sentence = np.tile(np.array([[[[PAD]]]], dtype=np.float32), [
                current_batch_size, 1,
                self.hp.Data.max_words_per_sentence_len,
                self.hp.Data.word_feature_num])

            for turn_num in range(1, turn_nums):
                DLG_HISTORY = np.concatenate(DLG[:turn_num * 2], axis=1)
                empty_matrix = np.tile(empty_sentence, [
                    1, self.hp.Data.max_turn_per_dialog_len * 2 - turn_num * 2, 1, 1])
                DLG_HISTORY = np.concatenate([DLG_HISTORY, empty_matrix], axis=1)
                LAST_QUE = np.reshape(DLG[turn_num * 2 - 2], [
                    current_batch_size,
                    self.hp.Data.max_words_per_sentence_len,
                    self.hp.Data.word_feature_num])
                QUE = np.reshape(DLG[turn_num * 2], [
                    current_batch_size,
                    self.hp.Data.max_words_per_sentence_len,
                    self.hp.Data.word_feature_num])
                ANS = np.reshape(DLG[turn_num * 2 + 1], [
                    current_batch_size,
                    self.hp.Data.max_words_per_sentence_len,
                    self.hp.Data.word_feature_num])
                ENID_QUE = np.reshape(ENID_DLG[turn_num * 2], [
                    current_batch_size,
                    self.hp.Data.max_words_per_sentence_len])
                LEN_QUE = np.reshape(LEN_DLG[turn_num * 2], [current_batch_size])
                RAW_QUE = [que[turn_num] for que in raw_list]

                if self.hp.project == 'origin':
                    yield (VGG, C3D, DLG_HISTORY, QUE, ANS, CAN[turn_num], ANS_ID[turn_num])
                else:
                    yield (VGG, C3D, DLG_HISTORY, LAST_QUE, ENID_QUE, LEN_QUE, RAW_QUE)

    def get_fake_batch_data(self):
        data = None
        for data in self.get_batch_data():
            break
        for _ in range(100000):
            yield data
