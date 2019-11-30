class HyperParams:
    epoch = 10
    middle_num_units = 128

    dropout_rate = 0.1
    learning_rate = 0.0001
    kernel_size = 5

    logdir = 'logdir'

    class Data:
        redis_host = 'localhost'
        redis_port = '1120'
        word_file = './Data/words.txt'
        word_embedding = './Data/word_embedding.pkl'
        train_records = 96000
        test_records = 2880

        max_words_per_sentence_len = 30
        max_turn_per_dialog_len = 15  # 1 + 13 + 1
        max_sentence_per_dialog_len = max_turn_per_dialog_len * 2
        word_feature_num = 100
        candidate_num = 50

        batch_size = 4

        vgg_frames = 80
        c3d_frames = 65
        vgg_feature_num = 4096
        c3d_feature_num = 4096
