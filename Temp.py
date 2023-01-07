import numpy as np
from torch.utils.data import Dataset
import torch, json
from torch import nn
import config
import torch.nn.functional as F
import config
import spacy
from utils.utils import *

nlp = spacy.load('en_core_web_sm')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


chars = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]"""
char_lst = [w for w in chars]
chars = ' '.join(char_lst)
tokens = nlp(chars)

# data = load_json(config.dev_file)
# context_lst, question_lst, ans_lst = print_samples(data)
#
# for context, question, an in zip(context_lst, question_lst, ans_lst):
#     context = [w.text for w in nlp(context, disable=['parser', 'tagger', 'ner'])]
#     question = [w.text for w in nlp(question, disable=['parser', 'tagger', 'ner'])]
#     answers = []
#     for ans in ans_lst:
#         answers.append([w.text for w in nlp(an, disable=['parse', 'tagger', 'ner'])] for an in ans if an is not None)
#
#     print('context is: ', ' '. join(context))
#     print('question is: ', ' '.join(question))
#     for answer in answers:
#         print('answer is: ', ' '.join(answer))
#

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


if __name__ == '__main__':
    # ques_lst = [[1, 1, 1, 1, 1, 1, 0, 0, 0]]
    # ctx_lst = [[1], [1], [1], [1], [1], [1], [0], [0], [0]]
    #
    # inputs = torch.ones(9, 9)
    # masked_inputs = inputs * torch.tensor(ctx_lst)
    # # print(masked_inputs)
    #
    # masked_inputs = masked_inputs * torch.tensor(ques_lst)
    # # print(masked_inputs)
    # sent_ids = [list(range(5)), list(range(4)), list(range(10)), list(range(5))]
    # sent_mask = _sent_mask(sent_ids)
    # print(sent_mask)
    #
    # token1 = ['my', ]

    file_path = './data/no_answer/train_df.pkl'
    df = pickle.load(open(file_path, 'rb'))
    print(111)

    from transformers import BertModel, BertConfig

    bert_dir = 'hfl/chinese-macbert-base'
    config = BertConfig.from_pretrained(bert_dir)
    # 所有层的特征都输出，不加这个就不输出所有层
    # config.update({'output_hidden_states': True})
    bert = BertModel.from_pretrained(bert_dir, config=config)




































