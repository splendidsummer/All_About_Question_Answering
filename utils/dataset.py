import torch
from torch.utils.data import Dataset, DataLoader
import os, pickle, pdb
import config
import nltk


class SquadDataset(Dataset):
    def __init__(self, context_ids, context_char_ids, question_ids,
                 question_char_ids, labels, ids, padding_idx=1, char_padding_idx=1):
        # super(SquadDataset, self).__init__()
        self.ids = ids
        self.context_ids = context_ids
        self.context_char_ids = context_char_ids
        self.question_ids = question_ids
        self.question_char_ids = question_char_ids
        self.labels = labels
        self.padding_idx = padding_idx
        self.char_padding_idx = char_padding_idx
        self.max_word_len = config.max_len_word
        self.some_att = None

    def __getitem__(self, index):
        context_id = self.context_ids[index]
        context_char_id = self.context_char_ids[index]
        question_id = self.question_ids[index]
        question_char_id = self.question_char_ids[index]
        label = self.labels[index]
        identity = self.ids[index]

        return context_id, context_char_id, question_id, question_char_id, label, identity

    def __len__(self):
        return len(self.context_ids)

    def batch_data_pro(self, batch_datas):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        context_ids = [i for (i, _, _, _, _, _) in batch_datas]
        context_lengths = self._get_seq_lengths(context_ids)
        context_char_ids = [i for (_, i, _, _, _, _) in batch_datas]
        question_ids = [i for (_, _, i, _, _, _) in batch_datas]
        question_lengths = self._get_seq_lengths(question_ids)
        question_char_ids = [i for (_, _, _, i, _, _) in batch_datas]
        labels = torch.tensor([i for (_, _, _, _, i, _) in batch_datas], dtype=torch.long, device=device)
        ids = [i for (_, _, _, _, _, i) in batch_datas]

        padded_context = torch.tensor(self._pad_sent(context_ids), dtype=torch.long, device=device)
        padded_context_char = torch.tensor(self._pad_char(context_char_ids), dtype=torch.long, device=device)

        padded_question = torch.tensor(self._pad_sent(question_ids), dtype=torch.long, device=device)
        padded_question_char = torch.tensor(self._pad_char(question_char_ids), dtype=torch.long, device=device)

        context_masks = torch.tensor(self._sent_mask(context_ids), dtype=torch.long, device=device)
        question_masks = torch.tensor(self._sent_mask(question_ids), dtype=torch.long, device=device)

        context_lengths = torch.tensor(context_lengths, dtype=torch.long, device=device)
        question_lengths = torch.tensor(question_lengths, dtype=torch.long, device=device)

        return padded_context, padded_context_char, padded_question, padded_question_char, context_masks, \
               question_masks, labels, context_lengths, question_lengths, ids

    @staticmethod
    def _sent_mask(sent_ids):
        sent_lens = [len(sent) for sent in sent_ids]
        max_len = max(sent_lens)
        masks = torch.zeros((len(sent_ids), max_len))
        for i, length in enumerate(sent_lens):
            masks[i, : length] = 1

        return masks

    @staticmethod
    def _get_seq_lengths(self, sent_ids):
        lengths = [len(sent) for sent in sent_ids]
        return lengths

    def _pad_sent(self, sent_ids):
        sent_lens = [len(sent) for sent in sent_ids]
        max_len = max(sent_lens)
        padded_sents = [sent + [self.padding_idx] * (max_len - len(sent)) for sent in sent_ids]
        # print(padded_sents)
        return padded_sents

    def _pad_char(self, char_ids):
        """
        :param char_ids:
        :param char_padding_idx:
        :param max_word_len: from the config setting
        :return:
        """
        max_len = max([len(sent) for sent in char_ids])

        padded_chars = [[w[: self.max_word_len] + [self.char_padding_idx] *
                         (self.max_word_len - len(w)) for w in s] for s in char_ids]
        dummy_word = [self.char_padding_idx] * self.max_word_len
        padded_chars = [sent + [dummy_word] * (max_len - len(sent)) for sent in padded_chars]

        return padded_chars


# if __name__ == '__main__':

    # context = [
    #     list(range(2)),
    #     list(range(3)),
    #     # list(range(8)),
    #     # list(range(12))
    # ]
    #
    # question = [
    #     list(range(3)),
    #     list(range(1)),
    #     # list(range(8)),
    #     # list(range(12))
    # ]
    #
    # context_char = [
    #     [list(range(5)), list(range(6))],
    #     [list(range(8)), list(range(12)), list(range(6))]
    # ]
    #
    # question_char = [
    #     [list(range(4)), list(range(8)), list(range(12))],
    #     [list(range(5))],
    #     # list(range(8)),
    #     # list(range(12))
    # ]
    #
    # labels = [1, 2]
    #
    # dataset = SquadDataset(context, context_char, question,
    #                        question_char, labels)
    #
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.batch_data_pro)
    #
    # for data_batch in dataloader:
    #     # assuming padded_context, padded_context_char, padded_question,
    #     # padded_question_char, context_masks, question_mask
    #     for item in data_batch:
    #         print(item)
