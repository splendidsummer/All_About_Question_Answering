import config
import wandb
import numpy as np
from utils.utils import *
from models.BIDAF import *
from torch import optim
import os, sys, time, tqdm, datetime, pickle, json
from utils.dataset import *
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
    device, n_gpu, args.fp16))

setup_seed(3407)
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
batch_size = config.batch_size
num_epochs = config.num_epochs
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

with open(config.idx2word_path, 'rb', encoding='utf-8') as f:
    idx2word = pickle.load(f)

with open(config.dev_df_path, 'rb', encoding='utf-8') as f:
    val_df = pickle.load(f)

val_ids = val_df.id

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
wandb.watch(model)

# optimizer Adadelta correction with the scale of gradients
optimizer = optim.Adadelta(model.parameters(),
                           lr=lr,
                           # weight_decay= config.weight_decay
                           )

loss_fn = nn.CrossEntropyLoss()


def evaluate_customize(predictions):

    pass


def evaluate(predictions):
    '''
    Gets a dictionary of predictions with question_id as key
    and prediction as value. The validation dataset has multiple
    answers for a single question. Hence we compare our prediction
    with all the answers and choose the one that gives us
    the maximum metric (em or f1).
    This method first parses the JSON file, gets all the answers
    for a given id and then passes the list of answers and the
    predictions to calculate em, f1.


    :param dict predictions
    Returns
    : exact_match: 1 if the prediction and ground truth
      match exactly, 0 otherwise.
    : f1_score:
    '''

    with open('./data/dev-v2.0.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data['data']   # dataset is a list object
    f1 = exact_match = total = 0
    for article in data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    continue

                ground_truths = list(map(lambda x: x['text'].lower(), qa['answers']))

                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


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

        context, question, char_ctx, char_ques, ctx_masks, ques_masks, labels, ctx_lens, ques_lens = batch
        preds = model(context, question, char_ctx, char_ques, ctx_masks, ques_masks, ctx_lens, ques_lens)
        start_pred, end_pred = preds
        s_idx, e_idx = labels[:, 0], labels[:, 1]
        loss = loss_fn(start_pred, s_idx) + loss_fn(end_pred, e_idx)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(trainset)


def valid_one_epoch():
    print("Starting validation .........")
    valid_loss = 0.
    batch_count = 0

    model.eval()
    predictions = []

    for batch in val_loader:
        ids = val_ids[batch_count*batch_size: (batch_count+1)*batch_size]
        if batch_count % 100 == 0:
            print(f"Starting batch {batch_count}")

        context, question, char_ctx, char_ques, context_mask, question_masks, labels, ctx_lens, ques_lens = batch

        with torch.no_grad():
            s_idx, e_idx = labels[:, 0], labels[:, 1]
            preds = model(context, question, char_ctx, char_ques, context_mask, question_masks, ctx_lens, ques_lens)
            p1, p2 = preds
            loss = loss_fn(p1, s_idx) + loss_fn(p2, e_idx)

            valid_loss += loss.item()

            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).\
                unsqueeze(0).expand(batch_size, -1, -1)

            # shape of ls(p1).unsequeeze(2) = [bs, c_len, 1],
            # shape of ls(p2).unsqueeze(1) = [bs, 1, c_len]
            # shape of the expression: [bs, c_len, c_len]
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                val_id = ids[i]
                pred = context[i][s_idx[i]:e_idx[i] + 1]
                pred = ' '.join([idx2word[idx.item()] for idx in pred])
                predictions[val_id] = pred

        batch_count += 1

    # 这里的问题是原始的answer里面是否去除了特殊字符， 是否都更改成了小写
    em, f1 = evaluate(predictions)
    return valid_loss / len(valset), em, f1


def train():
    train_losses = []
    valid_losses = []
    ems = []
    f1s = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        start_time = time.time()

        train_loss = train_one_epoch()
        valid_loss, em, f1 = valid_one_epoch()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
            'em': em,
            'f1': f1,
        }, 'bidaf_run_{}.pth'.format(epoch))

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        ems.append(em)
        f1s.append(f1)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': valid_loss,
                   'exact_match': em, 'f1_score': f1})

        print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
        print(f"Epoch valid loss: {valid_loss}")
        print(f"Epoch EM: {em}")
        print(f"Epoch F1: {f1}")
        print("====================================================================================")

    wandb.save(model)

    torch.save(model.state_dict(), 'model.pth')  # possible to use .h5 file here??
    wandb.save('model_' + now + '.pth')   # pth?? pt?? h5??
