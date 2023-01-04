import numpy as np
from torch.utils.data import Dataset
import torch, json
from torch import nn
import config
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
idx2word = {1: 'my', 2: 'book'}


###############################################################
# Debugging for start & end index predictions
################################################################
#
# batch_size, c_len = 2, 10
#
# # idx starting from 0, the 1st start_idx = 2, the 2nd start_idx = 6
# start_idxs = [
#     [2, 3, 10, 5, 3, 2, 4, 1, 4, 2],
#     [2, 3, 1, 5, 3, 2, 10, 1, 4, 2],
# ]
# p1 = torch.tensor(start_idxs, dtype=torch.float32)
#
# # idx starting from 0, the 1st end_idx = 5, the 2nd end_idx = 4
# end_idxs = [
#     [2, 3, 3, 5, 3, 20, 4, 1, 4, 2],
#     [2, 3, 1, 5, 20, 2, 4, 1, 10, 2],
# ]
#
# p2 = torch.tensor(end_idxs, dtype=torch.float32)
#
# ls = nn.LogSoftmax(dim=1)  # specify dimension in softmax
# mask = (torch.ones(c_len, c_len) * float('-inf')).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
#
# # shape of ls(p1).unsequeeze(2) = [bs, c_len, 1],
# # shape of ls(p2).unsqueeze(1) = [bs, 1, c_len]
# # shape of the expression: [bs, c_len, c_len]
# score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
# score, s_idx = score.max(dim=1)
# print('score is ', score, 'start index is ', s_idx)
# score, e_idx = score.max(dim=1)
# print('score is ', score, 'end index is ', e_idx)
# # shape of e_idx.view(-1, 1).shape  = [bs, 1]
# s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
# print('score is ', score, 'start index is ', s_idx)
# print(1111)
#


###############################################################
# Debugging for start & end index predictions
################################################################


# def evaluate(predictions):
def evaluate():
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
                # if qa['id'] not in predictions:
                #     continue

                ground_truths = list(map(lambda x: x['text'], qa['answers']))

                # prediction = predictions[qa['id']]

                # exact_match += metric_max_over_ground_truths(
                #     exact_match_score, prediction, ground_truths)
                #
                # f1 += metric_max_over_ground_truths(
                #     f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    # return exact_match, f1
    return ground_truths


ground_truths = evaluate()

print(ground_truths)


def _sent_mask(sent_ids):
    sent_lens = [len(sent) for sent in sent_ids]
    max_len = max(sent_lens)
    masks = torch.zeros((len(sent_ids), max_len))
    for i, length in enumerate(sent_lens):
        masks[i, : length] = 1

    return masks


def valid(model, valid_dataset):
    print("Starting validation .........")

    valid_loss = 0.

    batch_count = 0

    f1, em = 0., 0.

    model.eval()

    predictions = {}

    for batch in valid_dataset:

        if batch_count % 500 == 0:
            print(f"Starting batch {batch_count}")
        batch_count += 1

        context, question, char_ctx, char_ques, label, ctx, answers, ids = batch

        context, question, char_ctx, char_ques, label = context.to(device), question.to(device), \
                                                        char_ctx.to(device), char_ques.to(device), label.to(device)

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
                predictions[id] = [pred]

    em, f1 = evaluate(predictions)
    return valid_loss / len(valid_dataset), em, f1


if __name__ == '__main__':
    ques_lst = [[1, 1, 1, 1, 1, 1, 0, 0, 0]]
    ctx_lst = [[1], [1], [1], [1], [1], [1], [0], [0], [0]]

    inputs = torch.ones(9, 9)
    masked_inputs = inputs * torch.tensor(ctx_lst)
    # print(masked_inputs)

    masked_inputs = masked_inputs * torch.tensor(ques_lst)
    # print(masked_inputs)
    sent_ids = [list(range(5)), list(range(4)), list(range(10)), list(range(5))]
    sent_mask = _sent_mask(sent_ids)
    print(sent_mask)

    token1 = ['my', ]



































