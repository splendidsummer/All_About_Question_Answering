import torch
from torch.utils.data import Dataset, DataLoader
import os, pickle, pdb
import nltk
from xml.dom.minidom import parse


class Drugdataset(Dataset):
    def __init__(self, data, word_index, suffix_index):  # tag_index can be defined in other place
        super(Drugdataset, self).__init__()
        labels = ['B-drug', 'B-drug_n', 'B-brand', 'B-group', 'I-drug', 'I-drug_n', 'I-brand', 'I-group', 'O', '<PAD>']
        self.tag2index = {tag: idx for (idx, tag) in enumerate(sorted(labels))}

        self.data = data
        self.word_index = word_index
        self.suffix_index = suffix_index

        self.tag_index = {label: idx for (idx, label) in enumerate(sorted(labels))}
        # print('tag index:  ', self.tag_index)
        self.sents_tokens, self.sids = self.get_sent_tokens
        self.sents_tags = self.get_sent_tags

    def __getitem__(self, index):
        seq_tokens = self.sents_tokens[index]
        seq_tags = self.sents_tags[index]
        seq_tokens = [self.word_index.get(w, self.word_index['<UNK>']) for w in seq_tokens]
        seq_suf_tokens = [self.suffix_index.get(w, self.suffix_index['<UNK>']) for w in seq_tokens]

        seq_tags = [self.tag_index[tag] for tag in seq_tags]
        return seq_tokens, seq_suf_tokens, seq_tags

    def __len__(self):
        assert len(self.sents_tokens) == len(self.sents_tags)
        return len(self.sents_tags)

    @property
    def get_sent_tokens(self):
        sents_tokens = []
        sids = []
        for sid in self.data:
            sent_tok_lst = self.data[sid]
            sent_tokens = [sent_tok_lst[i]['form'] for i in range(len(sent_tok_lst))]
            sents_tokens.append(sent_tokens)
            sids.append(sid)
        return sents_tokens, sids

    @property
    def get_sent_tags(self):
        sents_tags = []
        for sid in self.data:
            sent_tok_lst = self.data[sid]
            sent_tags = [sent_tok_lst[i]['tag'] for i in range(len(sent_tok_lst))]
            sents_tags.append(sent_tags)
        return sents_tags

    @property
    def tokens(self):
        for sid in self.data:
            s = []
            for w in self.data[sid]:  # w = tokeb in upper part
                s.append((sid, w['form'], w['start'], w['end']))
            yield s

    def batch_data_pro(self, batch_datas):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_len = [len(data) for data, _, _ in batch_datas]
        if any in data_len == 0:
            pdb.set_trace()
        max_len = max(data_len)
        data = [i + [self.word_index['<PAD>']] * (max_len - len(i)) for (i, _, _) in batch_datas]
        data_suf = [i + [self.suffix_index['<PAD>']] * (max_len - len(i)) for (_, i, _) in batch_datas]
        # 这里有问题, tag的pad标签不对
        tags = [i + [self.tag2index['<PAD>']] * (max_len - len(i)) for (_, _, i) in batch_datas]
        data = torch.tensor(data, dtype=torch.long, device=device)
        data_suf = torch.tensor(data_suf, dtype=torch.long, device=device)
        tags = torch.tensor(tags, dtype=torch.long, device=device)
        return data, data_suf, tags, data_len


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    word_index = pickle.load(open('../data/preprocess/drug_vocab.pkl', 'rb'))
    parse_train_file = '../data/preprocess/parse_train_data.pkl'
    parse_devel_file = '../data/preprocess/parse_devel_data.pkl'
    parse_test_file = '../data/preprocess/parse_test_data.pkl'

    train_data = pickle.load(open(parse_train_file, 'rb'))
    devel_data = pickle.load(open(parse_devel_file, 'rb'))
    test_data = pickle.load(open(parse_test_file, 'rb'))

    dataset = Drugdataset(train_data, word_index)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.batch_data_pro)
    i = 0

    for data, tag, lens in dataloader:
        print('{}th data bacth'.format(i))

        try:
            print(data.shape)
            print(tag.shape)
            print(lens[0])


        except ValueError:
            print('Wrong')