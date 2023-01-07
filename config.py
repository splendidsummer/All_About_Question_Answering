import datetime

now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')

# System setting
TEAM_NAME = 'unicorn_upc_dl'

# experiment ID
exp = "_" + now

# data directories
train_file = '../data/train-v2.0.json'
dev_file = './data/dev-v2.0.json'
data_dir = "../data/"
full_data_dir = '../data/no_answer/'
glove_path = './data/glove.6B.100d.txt'

# model paths
spacy_en = "/root/miniconda3/lib/python3.8/site-packages/en_core_web_sm/en_core_web_sm-2.3.1"
glove = "./glove_embeddings/" + "glove.6B.{}d.txt"

# processed data file
glove_mat_path = './data/glove_matrix.pkl'
vocab_path = './data/vocab_word2idx.pkl'
char_vocab = './data/char2idx.pkl'

train_df_path = './data/train_df.pkl'
train_df_noanswer_path = './data/no_answer/train_df.pkl'
dev_df_path = './data/dev_df.pkl'
dev_df_noanswer_path = './data/no_answer/dev_df.pkl'

train_feature_path = './data/train_features.npz'
train_feature_noanswer_path = './data/no_answer/train_features.npz'

dev_feature_path = './data/dev_features.npz'
dev_feature_noanswer_path = './data/no_answer/dev_features.npz'

idx2word_path = './data/vocab_idx2word.pkl'
idx2word_noanswer_path = './data/no_answer/vocab_idx2word.pkl'
word2idx_path = './data/vocab_word2idx.pkl'
word2idx_noanswer_path = './data/no_answer/vocab_word2idx.pkl'

wrong_id_path = '。/data/no_answer/wrong_ids.txt'

# wandb configuration
max_len_word = 25
wandb_config = {
    'word_embedding_size': 100,
    'char_embedding_size': 8,
    'max_len_context': 400,
    'max_len_question': 50,
    'vocab_size': 97110,
    'char_vocab_size': 204,
    'max_len_word': max_len_word,
    'num_epochs': 12,  #
    'batch_size': 32,
    'learning_rate': 0.5,
    'drop_prob': 0.2,
    'hidden_size': 100,
    'charcnn_kernel_size': 5,
    # 'char_channel_size': 100,
    'cuda': True,
    'pretrained': False,
    'weight_decay': 0.0001,
}

# raw-vocab: 97108
# vocab-length: 97110
# word2idx-length: 97110
# raw-char-vocab: 1310
# char-vocab-intersect: 202
# char2idx-length: 204

# model saving path
squad_models = "output/" + exp

#  network params
char_vocab_size = 204

