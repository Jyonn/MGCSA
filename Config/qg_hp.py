from Config.hyperparams import HyperParams

qg_hp_TACoS = HyperParams()
qg_hp_TACoS.learning_rate = 0.0005
qg_hp_TACoS.dataset = 'TACoS'
qg_hp_TACoS.logdir = 'qg_logdir/%s' % qg_hp_TACoS.dataset
qg_hp_TACoS.project = 'qg'
qg_hp_TACoS.Data = HyperParams.Data()
qg_hp_TACoS.Data.batch_size = 4


qg_hp_YoutubeClip = HyperParams()
qg_hp_YoutubeClip.project = 'qg'
qg_hp_YoutubeClip.dataset = 'YoutubeClip'
qg_hp_YoutubeClip.logdir = 'qg_logdir/%s' % qg_hp_YoutubeClip.dataset
qg_hp_YoutubeClip.Data = HyperParams.Data()
qg_hp_YoutubeClip.Data.batch_size = 4
qg_hp_YoutubeClip.Data.word_file = './Data/YoutubeClip/words.txt'
qg_hp_YoutubeClip.Data.word_embedding = './Data/YoutubeClip/word_embedding.pkl'
qg_hp_YoutubeClip.Data.train_records = 160000
qg_hp_YoutubeClip.Data.test_records = 3000
qg_hp_YoutubeClip.Data.vocab_size = 5607
qg_hp_YoutubeClip.Data.vgg_frames = 60
qg_hp_YoutubeClip.Data.c3d_frames = 45
