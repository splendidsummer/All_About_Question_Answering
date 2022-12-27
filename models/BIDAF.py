import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_char, vocab,
                 dropout,drop_highway=0.3, bidirectional=True):
        super(Encoder, self).__init__()
        padding_idx = vocab['<pad>']
        vocab_size = len(vocab.keys())
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.charcnn = CharCNN(num_char, embed_size)
        self.highway = Highway(embed_size, drop_highway)
        self.encoder = nn.LSTM(embed_size, hidden_size,bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_char_tensor, input_token_tensor, seq_lengths):
        embed1 = self.embeddings(input_token_tensor)
        embed2 = self.charcnn(input_char_tensor)

        embed = torch.concat(embed1, embed2, )

        return out


class Decoder(nn.Module):
    pass


class CharCNN(nn.Module):
    def __init__(self, in_ch, out_ch, max_char_length, kernel_size=5):
        super(CharCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size)
        self.maxpool = nn.MaxPool1d(max_char_length-kernel_size+1)

    def forward(self, input_char_tensor):
        """
        :param input_char_tensor:
        :return:
        """
        conv_chars = self.conv(input_char_tensor)
        conv_chars = self.maxpool(F.relu(conv_chars))

        return conv_chars


class Highway(nn.Module):
    def __init__(self, embed_size, drop_prob):
        super(Highway, self).__init__()
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, inputs):
        out = self.linear1(inputs).clip(min=0)
        out = self.linear2(out)
        gate = self.sigmoid(out)
        out = out * gate  + (1-gate) * inputs
        out = self.dropout(out)
        return out


class BiDAF(nn.Module):
    def __init__(self, embed_size, hidden_size, num_char, vocab,
                 bidirectional=True ):
        pass

    def forward(self):
        pass


