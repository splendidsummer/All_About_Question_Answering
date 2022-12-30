# experiment ID
exp = "exp-0"

# data directories
train_file = '../data/train-v2.0.json'
dev_file = '../data/dev-v2.0.json'
data_dir = "./data/"
glove_path = './data/glove.6B.100d.txt'

# model paths
spacy_en = "/root/miniconda3/lib/python3.8/site-packages/en_core_web_sm/en_core_web_sm-2.3.1"
glove = "./glove_embeddings/" + "glove.6B.{}d.txt"
squad_models = "output/" + exp

# preprocessing values
max_words = -1
word_embedding_size = 100
char_embedding_size = 8
max_len_context = 400
max_len_question = 50
max_len_word = 25

# training hyper-parameters
num_epochs = 1
batch_size = 1
learning_rate = 0.5
drop_prob = 0.2
hidden_size = 100
char_channel_width = 5
char_channel_size = 100
cuda = True
cuda = False
pretrained = False
