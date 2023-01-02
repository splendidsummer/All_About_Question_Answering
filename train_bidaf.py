import config
import wandb
import numpy as np
from utils.utils import *
from models.BIDAF import *
from torch import optim
import os, sys, time, tqdm, datetime, pickle, json
from utils.dataset import *

setup_seed(168)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')
model_path = '/root/autodl-tmp/dl_project2/saved_model_' + now + '.pt'

# initialize wandb logging to your project
wandb.init(
    job_type='Question_Answering_Squad2.0',
    project="BiDAF_model",
    dir='/root/autotmp-dl/All_About_Question_Answering/wandb',
    entity=config.TEAM_NAME,
    config=config.wandb_config,
    # sync_tensorboard=True,
    name='BiDAF Training',
    notes='min_lr=0.00001',
    ####
)

config = wandb.config
batch_size  = config.batch_size
epochs = config.freeze_epochs
lr = config.learning_rate
weight_decay = config.weight_decay
activation = config.activation
embed_size = config.word_embedding_size
char_embed_size = config.char_embedding_size
max_word_length = config.max_word_length
hidden_size = config.hidden_size
vocab_size = config.vocab_size
char_vocab_size = config.char_vocab_size
drop_rate = config.drop_rate


with open(config.glove_mat_path, 'rb', encoding='utf-8') as f:
    glove_vectors = pickle.load(f)

with open(config.glove_path, 'rb', encoding='utf-8') as f:
    char_vectors = pickle.load(f)

train_data = np.load(config.train_feature_path)  # using like a dict
dev_data = np.load(config.dev_feature_path)

train_data = [i for (_, i) in train_data.items()]
dev_data = [i for (_, i) in dev_data.items()]

trainset = SquadDataset(*train_data)
valset = SquadDataset(*dev_data)
train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=trainset.batch_data_pro
                          )

val_loader = DataLoader(valset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=valset.batch_data_pro
                        )


model = BiDAF(glove_vectors, char_vocab_size, char_embed_size, embed_size,
              hidden_size, max_word_length, drop_rate).to(device)
.
wandb.watch(model)

# optimizer Adadelta correction with the scale of gradients
optimizer = optim.Adadelta(model.parameters(),
                           lr=lr,
                           # weight_decay= config.weight_decay
                           )

loss_fn = nn.CrossEntropyLoss()


def train_one_epoch():
    print("Starting training ........")
    train_loss = 0.
    batch_count = 0
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        if batch_count % 500 == 0:
            print(f"Starting batch: {batch_count}")
        batch_count += 1

        context, question, char_ctx, char_ques, ctx_masks, ques_mask, labels = batch
        preds = model(context, question, char_ctx, char_ques, ctx_masks, ques_mask)
        start_pred, end_pred = preds
        s_idx, e_idx = labels[:, 0], labels[:, 1]
        loss = loss_fn(start_pred, s_idx) + loss_fn(end_pred, e_idx)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(trainset)


def valid_one_epoch(model, valid_dataset):
    print("Starting validation .........")
    valid_loss = 0.
    batch_count = 0
    # f1, em = 0., 0.

    model.eval()
    predictions = {}

    for batch in valid_dataset:

        if batch_count % 100 == 0:
            print(f"Starting batch {batch_count}")
        batch_count += 1

        context, question, char_ctx, char_ques, label, ctx, answers, ids = batch
        # context, question, char_ctx, char_ques, label = context.to(device), question.to(device), \
        #                                                 char_ctx.to(device), char_ques.to(device), label.to(device)

        with torch.no_grad():

            s_idx, e_idx = label[:, 0], label[:, 1]

            preds = model(context, question, char_ctx, char_ques)

            p1, p2 = preds

            loss = F.cross_entropy(p1, s_idx) + F.cross_entropy(p2, e_idx)

            valid_loss += loss.item()

            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1,
                                                                                                      -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                id = ids[i]
                pred = context[i][s_idx[i]:e_idx[i] + 1]
                pred = ' '.join([idx2word[idx.item()] for idx in pred])
                predictions[id] = pred

    em, f1 = evaluate(predictions)
    return valid_loss / len(valid_dataset), em, f1


wandb.log({'epoch': epoch, 'train_loss': loss})

wandb.save(model)

torch.save(model.state_dict(), 'model.pth')  # possible to use .h5 file here??
wandb.save('model_' + now + '.pth')   # pth?? pt?? h5??
