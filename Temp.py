import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn
import config

batch_size, c_len = 2, 10

# idx starting from 0, the 1st start_idx = 2, the 2nd start_idx = 6
start_idxs = [
    [2, 3, 10, 5, 3, 2, 4, 1, 4, 2],
    [2, 3, 1, 5, 3, 2, 10, 1, 4, 2],
]
p1 = torch.tensor(start_idxs, dtype=torch.long)

# idx starting from 0, the 1st end_idx = 5, the 2nd end_idx = 4
end_idxs = [
    [2, 3, 3, 5, 3, 20, 4, 1, 4, 2],
    [2, 3, 1, 5, 20, 2, 4, 1, 10, 2],
]

p2 = torch.tensor(end_idxs, dtype=torch.long)

ls = nn.LogSoftmax(dim=1)  # specify dimension in softmax
mask = (torch.ones(c_len, c_len) * float('-inf')).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)

score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
score, s_idx = score.max(dim=1)
print('score is ', score, 'start index is ', s_idx)
score, e_idx = score.max(dim=1)
print('score is ', score, 'end index is ', e_idx)

s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
print('score is ', score, 'start index is ', s_idx)


#
# def exact_sy_features(sent):
#     doc = nlp(sent)
#     tokens = [[token.text, token.pos_, token.tag_, token.lemma_, str(token.is_stop), token.dep_
#                # token.ent_type_, token.dep_
#                ] for token in doc if '\n' not in token.text and '\r' not in token.text]
#     # tokens = [[token.text, str(token.is_stop)] for token in doc if '\n' not in token.text and '\r' not in token.text]
#     return tokens
#


# if __name__ == '__main__':