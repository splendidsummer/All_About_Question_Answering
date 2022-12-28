import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):
    def __init__(self, glove_vectors, char_vectors, embed_size, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        assert glove_vectors.size(1) == embed_size, 'pretrained wording embedding size conflicts with designated embedding size.'
        self.wembed = nn.Embedding.from_pretrained(glove_vectors, freeze=True)
        self.cembed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.proj =


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_char, max_char_length,
                 vocab, dropout, drop_highway=0.3, bidirectional=True):
        super(Encoder, self).__init__()
        padding_idx = vocab['<pad>']
        vocab_size = len(vocab.keys())
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.charcnn = CharCNN(num_char, embed_size, max_char_length)
        self.highway = Highway(embed_size*2, drop_highway)
        self.encoder = nn.LSTM(embed_size, hidden_size,bidirectional=bidirectional)
        # self.projection = nn.Linear(hidden_size*2, hidden_size)  # here out dim is hidden_size ????
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_char_tensor, input_token_tensor, seq_lengths):

        """
        :param input_char_tensor:
        :param input_token_tensor: shape = (seq_length, batch_size)
        :param seq_lengths:
        :return:
        """

        embed1 = self.embeddings(input_token_tensor)
        embed2 = self.charcnn(input_char_tensor)
        embed = torch.concat([embed1, embed2], dim=-1)

        out = self.highway(embed)
        out = pack_padded_sequence(out, seq_lengths)
        hiddens, (last_hidden, last_cell) = self.encoder(out)
        hiddens = pad_packed_sequence(hiddens, batch_first=True, total_length=input_token_tensor.size(0))
        hiddens = hiddens[0]

        return hiddens


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
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


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, inputs, attention_mask):
        pass


class idontknow(nn.Module):
    pass


class BiDAF(nn.Module):
    def __init__(self, embed_size, hidden_size, num_char, vocab,
                 bidirectional=True):
        self.context_encoder = Encoder()
        self.query_encoder = Encoder()
        pass

    def generate_sent_masks(self, encodings, seq_lengths):
        masks = torch.zeros(encodings.size(0), encodings.size(1), dtype=torch.float)
        for i, idx in enumerate(seq_lengths):
            masks[i, idx:] = 1
        # print(masks.device)
        # 这里是否需要转移mask到device上面?????

    def forward(self, input_context, input_query, input_context_chars, input_query_chars):
        pass


