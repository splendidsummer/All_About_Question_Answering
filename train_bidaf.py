import config
import wandb
from utils.utils import *
from models.BIDAF import *
from torch import optim
import os, sys, time, tqdm, datetime, pickle, json


setup_seed(168)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')
model_path = '/root/autodl-tmp/dl_project2/saved_model_' + now + '.pt'

# initialize wandb logging to your project
wandb.init(
    job_type='Question_Answering_Squad2.0',
    project="BiDAF_model",
    dir='/root/autotmp-dl/????',
    entity=config.TEAM_NAME,
    config=config.wandb_config,
    # sync_tensorboard=True,
    name='?????????',
    notes='min_lr=0.00001',
    ####
)

config = wandb.config
batch_size = config.batch_size
epochs = config.freeze_epochs
lr = config.learning_rate
weight_decay = config.weight_decay
# early_stopping = config.early_stopping
activation = config.activation
# augment = config.augment

with open(config.glove_mat_path, 'rb', encoding='utf-8') as f:
    glove_vectors = pickle.load(f)

with open(config.glove_path, 'rb', encoding='utf-8') as f:
    glove_vectors = pickle.load(f)


model = BiDAF(glove_vectors, char_vectors, embed_size, char_embed_size,
                hidden_size, vocab_size, max_word_length, drop_rate,
                 bidirectional=True)


# optimizer Adadelta correction with the scale of gradients
optimizer = optim.Adadelta(model.p

                            lr
)


# To log wandb
for epoch in range(10):
    wandb.log({'epoch': epoch, 'loss': loss})

    # by default, this will save to a new subfolder for files associated
    # with your run, created in wandb.run.dir (which is ./wandb by default)
    wandb.save("mymodel.h5")

config = wandb.config
batch_size = config.batch_size
num_epochs = config.num_epochs
lr = config.learning_rate
weight_decay = config.weight_decay
activation = config.activation

model = BiDAF(CHAR_VOCAB_DIM,
              EMB_DIM,
              CHAR_EMB_DIM,
              NUM_OUTPUT_CHANNELS,
              KERNEL_SIZE,
              HIDDEN_DIM,
              device).to(device)

model = BiDAF(glove_vectors, char_vectors, embed_size, char_embed_size,
                hidden_size, vocab_size, max_word_length, drop_rate,
                 bidirectional=True)


wandb.save(model)

torch.save(model.state_dict(), 'model.pth')  # possible to use .h5 file here??
wandb.save('model_' + now + '.pth')   # pth?? pt?? h5??
