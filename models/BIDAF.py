import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
    def __init__(self, num_layers, embed_size):
        super(Highway, self).__init__()
        self.transformers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_layers)])

    def forward(self, inputs):
        for gate, transform in zip(self.gates, self.transformers):
            transform_input = F.relu(transform(inputs))
            gate_value = torch.sigmoid(gate(inputs))
            inputs = inputs * (1 - gate_value) + gate_value * transform_input

        return inputs


class Embedding(nn.Module):
    def __init__(self, glove_vectors, char_vectors,
                 char_embed_size, embed_size, hidden_size, max_word_length,
                 drop_prob):

        super(Embedding, self).__init__()
        assert glove_vectors.size(1) == embed_size, 'pretrained wording embedding size' \
                                                    ' conflicts with designated embedding size.'

        self.wembed = nn.Embedding.from_pretrained(glove_vectors, freeze=True)
        self.cembed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.proj = nn.Linear(embed_size, hidden_size, bias=False)
        self.cnn = CharCNN(char_embed_size, embed_size, max_word_length)
        self.dropout = nn.Dropout(drop_prob)
        self.highway = Highway(2, embed_size)  # using 2 highway layers

    def forward(self, word_tensors, char_tensors):

        word_embed = self.wembed(word_tensors)
        word_embed = self.dropout(word_embed)

        char_embed = self.cembed(char_tensors)
        char_embed = self.cnn(char_embed)

        embed = torch.concat([word_embed, char_embed], dim=-1)
        embed = self.highway(embed)

        return embed


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, drop_rate):
        """
        :param input_size:
        :param hidden_size:
        :param drop_rate: if more than 1 layer of LSTM, we may need dropout
        :param bidirectional:
        """
        super(Encoder, self).__init__()
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, inputs, seq_lengths):
        """
        :param inputs: should be sorted sequence list w.r.t. corresponding length
        :param seq_lengths: length w.r.t. element of inputs
        :return:
        """

        total_len = inputs.size(1)
        out = pack_padded_sequence(inputs, seq_lengths)
        hiddens, _ = self.encoder(out)
        hiddens = pad_packed_sequence(hiddens, batch_first=True, total_length=total_len)
        hiddens = self.dropout(hiddens)

        return hiddens


class Attention(nn.Module):
    def __init__(self, hidden_size, drop_rate):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size*3, 1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, context, context_mask, query, query_mask):
        context = self.dropout(context)
        query = self.dropout(query)

        batch_size, context_len, query_len = context.size(0), context.size(1), query.size(1)

        context_ = context.unsequeeze(2).repeat(1, 1, query_len, 1)
        query_ = query.unsequeeze(1).repeat(1, context_len, 1, 1)

        elementwise_prod = torch.mul(context_, query_)

        # [bs, context_len, query_len, 6*hidden_size]
        cq = torch.cat([context_, query_, elementwise_prod], dim=-1)

        # sim_matrix.shape = [bs, context_len, query_len]
        sim_matrix = self.attention(cq).view(-1, context_len, query_len)
        alpha = F.softmax(sim_matrix, dim=-1)

        # [bs, context_len, query_len] * [bs, query_len, embed_size] ->
        # [bs, context_len, embed_size]
        a = torch.bmm(alpha, query)
        q2c_sim_mat = torch.max(sim_matrix, dim=-1)  # maybe there is a problem??
        # beta.shape = [bs, 1, context_len]
        beta = F.softmax(q2c_sim_mat, dim=-1).unsqueeze(1)
        b = torch.bmm(beta, context).repeat(1, context_len, 1)

        global_hidden = torch.concat([context, a, torch.mul(context, a), torch.mul(context, b)])

        return global_hidden


class ModelingLayer(nn.Module):
    def __init__(self, hidden_size, drop_rate):
        super(ModelingLayer, self).__init__()
        self.rnn = nn.LSTM(hidden_size*8, hidden_size, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=drop_rate)

    def forward(self, inputs):
        out, _ = self.rnn(inputs)
        return out


class Output(nn.Module):
    def __init__(self, hidden_size, drop_rate):
        super(Output, self).__init__()
        self.output_start = nn.Linear(hidden_size*10, 1, bias=False)
        self.output_end = nn.Linear(hidden_size*10, 1, bias=False)
        self.end_lstm = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, global_hidden, modeling_out):
        # Here the inputs are actually come from outputs of modeling layer
        # Since bidirectional=True, out shape == [bs, context_len, hidden_size*2]
        out, _ = self.end_lstm(global_hidden)
        start_pos = self.output_start(torch.cat([global_hidden, modeling_out], dim=-1)).squeeze()
        end_pos = self.output_end(torch.cat([global_hidden, out], dim=-1)).squeeze()

        return start_pos, end_pos


#############################################################
# components of BIDAF
# 1. Build embedding, load/initialize pretrained weights
# 2. Build context encoder,
#############################################################


class BiDAF(nn.Module):
    def __init__(self, glove_vectors, char_vectors, embed_size, char_embed_size,
                 hidden_size, vocab_size, max_word_length, drop_rate,
                 bidirectional=True):

        super(BiDAF, self).__init__()

        self.context_embeddings = Embedding(glove_vectors, char_vectors, char_embed_size,
                                    embed_size, hidden_size, max_word_length, drop_rate)
        self.query_embeddings = Embedding(glove_vectors, char_vectors, char_embed_size,
                                    embed_size, hidden_size, max_word_length, drop_rate)

        self.context_encoder = Encoder(embed_size, hidden_size, drop_rate)
        self.query_encoder = Encoder(embed_size, hidden_size, drop_rate)

        self.attention = Attention(hidden_size, drop_rate)

        self.modeling = ModelingLayer(hidden_size, drop_rate)

        self.output = Output(hidden_size, drop_rate)

    def forward(self, context_words, context_chars, context_masks, context_lens,
                      query_words, query_chars, query_masks, query_lens):

        context_embs = self.context_embeddings(context_words, context_chars)
        query_embs = self.query_embeddings(query_words, query_chars)

        context_encodes = self.context_encoder(context_embs, context_lens)
        query_encodes = self.query_encoder(query_embs, query_lens)

        global_hidden = self.attention(context_encodes, context_masks, query_encodes, query_masks)

        modeling_out = self.modeling(global_hidden)

        start_pos, end_pos = self.output(global_hidden, modeling_out)

        return start_pos, end_pos


class CharacterEmbeddingLayer(nn.Module):

    def __init__(self, char_vocab_dim, char_emb_dim, num_output_channels, kernel_size):
        super().__init__()

        self.char_emb_dim = char_emb_dim

        self.char_embedding = nn.Embedding(char_vocab_dim, char_emb_dim, padding_idx=1)

        self.char_convolution = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=kernel_size)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x = [bs, seq_len, word_len]
        # returns : [batch_size, seq_len, num_output_channels]
        # the output can be thought of as another feature embedding of dim 100.

        batch_size = x.shape[0]

        x = self.dropout(self.char_embedding(x))
        # x = [bs, seq_len, word_len, char_emb_dim]

        # following three operations manipulate x in such a way that
        # it closely resembles an image. this format is important before
        # we perform convolution on the character embeddings.

        x = x.permute(0, 1, 3, 2)
        # x = [bs, seq_len, char_emb_dim, word_len]

        x = x.view(-1, self.char_emb_dim, x.shape[3])
        # x = [bs*seq_len, char_emb_dim, word_len]

        x = x.unsqueeze(1)
        # x = [bs*seq_len, 1, char_emb_dim, word_len]

        # x is now in a format that can be accepted by a conv layer.
        # think of the tensor above in terms of an image of dimension
        # (N, C_in, H_in, W_in).

        x = self.relu(self.char_convolution(x))
        # x = [bs*seq_len, out_channels, H_out, W_out]

        x = x.squeeze()
        # x = [bs*seq_len, out_channels, W_out]

        x = F.max_pool1d(x, x.shape[2]).squeeze()
        # x = [bs*seq_len, out_channels, 1] => [bs*seq_len, out_channels]

        x = x.view(batch_size, -1, x.shape[-1])
        # x = [bs, seq_len, out_channels]
        # x = [bs, seq_len, features] = [bs, seq_len, 100]

        return x


if __name__ == '__main__':
    print(1111)



