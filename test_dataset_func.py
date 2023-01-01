import torch

padding_idx = 1
char_padding_idx = 1
max_word_len = 20


def _pad_sent(sent_ids):
    sent_lens = [len(sent) for sent in sent_ids]
    max_len = max(sent_lens)
    padded_sents = [sent + [padding_idx] * (max_len - len(sent)) for sent in sent_ids]
    # print(padded_sents)
    return padded_sents


def _pad_char(char_ids):
    """
    :param char_ids:
    :param char_padding_idx:
    :param max_word_len: from the config setting
    :return:
    """
    max_len = max([len(sent) for sent in char_ids])

    padded_chars = [[w[: max_word_len] + [char_padding_idx] *
                     (max_word_len - len(w)) for w in s] for s in char_ids]
    dummy_word = [char_padding_idx] * max_word_len
    padded_chars = [sent + [dummy_word] * (max_len - len(sent)) for sent in padded_chars]

    return padded_chars


if __name__ == '__main__':
    # padded_sents = [list(range(3)), list(range(6)), list(range(10))]
    char_ids = [[list(range(3)), list(range(4))], [list(range(3)), list(range(6)), list(range(10))]]
    char_ids = _pad_char(char_ids)
    char_ids = torch.tensor(char_ids, dtype=torch.long)
    print(char_ids)


    # print(_pad_sent(padded_sents))