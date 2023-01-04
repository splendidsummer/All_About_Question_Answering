import datetime

now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')

# System setting
TEAM_NAME = 'unicorn_upc_hle'

# experiment ID
exp = "_" + now

# data directories
train_file = '../data/train-v2.0.json'
dev_file = './data/dev-v2.0.json'
data_dir = "../data/"
glove_path = './data/glove.6B.100d.txt'

# model paths
spacy_en = "/root/miniconda3/lib/python3.8/site-packages/en_core_web_sm/en_core_web_sm-2.3.1"
glove = "./glove_embeddings/" + "glove.6B.{}d.txt"

# processed data file
glove_mat_path = './data/glove_embedding_matrix.pkl'
vocab_path = './data/vocab_word2idx.pkl'
char_vocab = './data/char2idx.pkl'
train_df_path = './data/train_df.pkl'
val_df_path = './data/dev_df.pkl'
train_feature_path = './data/train_features.npz'
dev_feature_path = './data/dev_features.npz'
idx2word_path = './data/vocab_idx2word.pkl'
word2idx_path = './data/vocab_word2idx.pkl'

# wandb configuration
max_len_word = 25
wandb_config = {
    'word_embedding_size': 100,
    'char_embedding_size': 8,
    'max_len_context': 400,
    'max_len_question': 50,
    'vocab_size': 94386,
    'char_vocab_size': 204,
    'max_len_word': max_len_word,
    'num_epochs': 12,  #
    'batch_size': 60,
    'learning_rate': 0.5,
    'drop_prob': 0.2,
    'hidden_size': 100,
    'charcnn_kernel_size': 5,
    # 'char_channel_size': 100,
    'cuda': True,
    'pretrained': False,
}

# model saving path
squad_models = "output/" + exp

#  network params
char_vocab_size = 204

