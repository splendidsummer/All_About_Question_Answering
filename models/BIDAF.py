import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config


class CharCNN(nn.Module):
    def __init__(self, out_ch, max_word_length, char_embed_size, kernel_size=5):
        super(CharCNN, self).__init__()
        self.max_word_length = max_word_length
        # applying kernel_size on sequence length
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_ch, kernel_size=(char_embed_size, kernel_size))

    def forward(self, input_char_tensor):
        """
        :param input_char_tensor:
        :return:
        """
        conv_chars = self.conv(input_char_tensor)
        return conv_chars


class Highway(nn.Module):
    def __init__(self, num_layers, embed_size):
        super(Highway, self).__init__()
        self.transformers = nn.ModuleList([nn.Linear(2*embed_size, 2*embed_size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(2*embed_size, 2*embed_size) for _ in range(num_layers)])

    def forward(self, inputs):
        for gate, transform in zip(self.gates, self.transformers):
            transform_input = F.relu(transform(inputs))
            gate_value = torch.sigmoid(gate(inputs))
            inputs = inputs * (1 - gate_value) + gate_value * transform_input

        return inputs


class Embedding(nn.Module):
    def __init__(self, glove_vectors, char_vocab_size, char_embed_size,
                 embed_size, hidden_size, max_word_length, drop_prob):

        super(Embedding, self).__init__()
        assert glove_vectors.size(1) == embed_size, 'pretrained wording embedding size' \
                                                    ' conflicts with designated embedding size.'

        self.char_embed_size = char_embed_size
        self.embed_size = embed_size

        self.wembed = nn.Embedding.from_pretrained(glove_vectors, freeze=True)
        self.cembed = nn.Embedding(config.char_vocab_size, char_embed_size, padding_idx=1)
        self.proj = nn.Linear(embed_size, hidden_size, bias=False)
        self.cnn = CharCNN(embed_size, max_word_length, char_embed_size)

        self.dropout = nn.Dropout(drop_prob)
        self.highway = Highway(2, embed_size)  # using 2 highway layers

    def forward(self, word_tensors, char_tensors):
        batch_size = char_tensors.size(0)
        word_embed = self.wembed(word_tensors)
        word_embed = self.dropout(word_embed)
        word_embed = self.proj(word_embed)

        char_embed = self.cembed(char_tensors)
        char_embed = self.dropout(char_embed)

        # char_embed.shape = [bs*seq_len, char_embed_size=8, max_word_len]
        char_embed = char_embed.view(-1, self.char_embed_size, char_embed.size(2)).unsqueeze(1)
        # char_embed.shape = [bs*seq_len, out_channels=100, 1, max_word_len-kernel_size+1]
        char_embed = self.cnn(char_embed)
        # char_embed.shape = [bs*seq_len, out_channels=100, max_word_len-kernel_size+1]
        char_embed = char_embed.squeeze()

        # shape before squeeze = [bs*seq_len, out_channels=100, 1], after squeeze = [bs*seq_len, out_channels=100]
        char_embed = F.max_pool1d(char_embed, char_embed.size(2)).squeeze()
        # char_embed.shape = [bs, seq_len, out_channels=100]
        char_embed = char_embed.view(batch_size, -1, self.embed_size)
        # embed.shape = [bs, seq_len, embed_size*2]
        embed = torch.concat([word_embed, char_embed], dim=-1)
        # embed.shape = [bs, seq_len, embed_size]
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
        self.encoder = nn.LSTM(2*embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, inputs, seq_lengths):
        """
        :param inputs: should be sorted sequence list w.r.t. corresponding length
        :param seq_lengths: length w.r.t. element of inputs
        :return:
        """

        total_len = inputs.size(1)
        lengths, sort_idx = seq_lengths.sort(0, descending=True)
        inputs = inputs[sort_idx]
        out = pack_padded_sequence(inputs, lengths, batch_first=True)
        out, _ = self.encoder(out)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=total_len)
        _, unsort_idx = sort_idx.sort(0)
        out = out[unsort_idx]

        out = self.dropout(out)

        return out


class Attention(nn.Module):
    def __init__(self, hidden_size, drop_rate):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size*6, 1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, context, context_masks, query, query_masks):
        context = self.dropout(context)
        query = self.dropout(query)

        # context_masks = context_masks.squeeze(-1)
        # question_masks = query_masks.squeeze(1)

        batch_size, context_len, query_len = context.size(0), context.size(1), query.size(1)

        context_ = context.unsqueeze(2).repeat(1, 1, query_len, 1)
        query_ = query.unsqueeze(1).repeat(1, context_len, 1, 1)

        elementwise_prod = torch.mul(context_, query_)

        # [bs, context_len, query_len, 6*hidden_size]
        cq = torch.cat([context_, query_, elementwise_prod], dim=-1)

        # sim_matrix.shape = [bs, context_len, query_len]
        sim_matrix = self.attention(cq).view(-1, context_len, query_len)

        alpha = F.softmax(sim_matrix, dim=-1)
        # print('alpha shape before masking', alpha.shape)

        # content_masks.squeeze(2).shape = [bs, ctx_len, 1]
        context_masks = context_masks.unsqueeze(-1)
        # question_masks.squeeze(1).shape = [bs, 1, ques_len]
        question_masks = query_masks.unsqueeze(1)
        # We should try using this alpha term with or without masking
        alpha = alpha * context_masks * question_masks

        # print('alpha shape before masking', alpha.shape)

        # [bs, context_len, query_len] * [bs, query_len, embed_size] ->
        # [bs, context_len, embed_size]
        a = torch.bmm(alpha, query)
        q2c_sim_mat, _ = torch.max(sim_matrix, dim=-1)  # maybe there is a problem??
        # beta.shape = [bs, 1, context_len]
        beta = F.softmax(q2c_sim_mat, dim=-1)
        #  [bs, 1, ctx_len] * [bs, 1, ctx_len]
        # We should try using this beta term with or without masking
        beta = beta * context_masks.squeeze()
        beta = beta.unsqueeze(1)

        # beta.shape = [bs, 1, context_len]
        b = torch.bmm(beta, context).repeat(1, context_len, 1)
        c = torch.mul(context, a)
        d = torch.mul(context, b)
        global_hidden = torch.concat([context, a, c, d], dim=-1)

        return global_hidden


class ModelingLayer(nn.Module):
    def __init__(self, hidden_size, drop_rate):
        super(ModelingLayer, self).__init__()
        self.rnn = nn.LSTM(hidden_size*8, hidden_size, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=drop_rate)

    def forward(self, inputs):
        out, _ = self.rnn(inputs)
        return out


def super(Output, self):
    pass


class Output(nn.Module):
    def __init__(self, hidden_size):
        super(Output, self).__init__()
        self.output_start = nn.Linear(hidden_size*10, 1, bias=False)
        self.output_end = nn.Linear(hidden_size*10, 1, bias=False)
        self.end_lstm = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, global_hidden, modeling_out):
        # Here the inputs are actually come from outputs of modeling layer
        # Since bidirectional=True, out shape == [bs, context_len, hidden_size*2]
        out, _ = self.end_lstm(modeling_out)
        start_pos = self.output_start(torch.cat([global_hidden, modeling_out], dim=-1)).squeeze()
        end_pos = self.output_end(torch.cat([global_hidden, out], dim=-1)).squeeze()

        return start_pos, end_pos


#############################################################
# components of BIDAF
# 1. Build embedding, load/initialize pretrained weights
# 2. Build context encoder,
#############################################################


class BiDAF(nn.Module):
    def __init__(self, glove_vectors, char_vocab_size, char_embed_size, embed_size,
                 hidden_size, max_word_length, drop_rate):

        super(BiDAF, self).__init__()

        self.context_embeddings = Embedding(glove_vectors, char_vocab_size, char_embed_size,
                                            embed_size, hidden_size, max_word_length, drop_rate)
        self.query_embeddings = Embedding(glove_vectors, char_vocab_size, char_embed_size,
                                          embed_size, hidden_size, max_word_length, drop_rate)

        self.context_encoder = Encoder(embed_size, hidden_size, drop_rate)
        self.query_encoder = Encoder(embed_size, hidden_size, drop_rate)

        self.attention = Attention(hidden_size, drop_rate)

        self.modeling = ModelingLayer(hidden_size, drop_rate)

        self.output = Output(hidden_size)

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


if __name__ == '__main__':
    batch_size, seq_len, qseq_len, max_word_len, char_embed_size = 4, 15, 10, 10, 8
    lens = torch.tensor([3, 5, 6, 8], dtype=torch.long)
    qlens = torch.tensor([7, 3, 5, 8])
    # ques_lens = [4, 7, 2, ]

    inputs = torch.randn(batch_size, seq_len, max_word_len, char_embed_size)
    qinputs = torch.randn(batch_size, qseq_len, max_word_len, char_embed_size)

    inputs = inputs.view(-1, char_embed_size, inputs.size(2)).unsqueeze(1)
    qinputs = qinputs.view(-1, char_embed_size, qinputs.size(2)).unsqueeze(1)

    charcnn = CharCNN(out_ch=100, max_word_length=max_word_len, char_embed_size=char_embed_size)

    out = charcnn(inputs)
    qout = charcnn(qinputs)

    out = out.squeeze()
    qout = qout.squeeze()

    # shape before squeeze = [bs*seq_len, out_channels=100, 1], after squeeze = [bs*seq_len, out_channels=100]
    out = F.max_pool1d(out, out.size(2)).squeeze()
    qout = F.max_pool1d(qout, qout.size(2)).squeeze()
    # char_embed.shape = [bs, seq_len, out_channels=100]
    out = out.view(batch_size, -1, 100)
    qout = qout.view(batch_size, -1, 100)
    out = torch.cat([out, out], dim=-1)
    qout = torch.cat([qout, qout], dim=-1)

    highway = Highway(2, 100)
    out = highway(out)
    qout = highway(qout)

    encoder = Encoder(100, 100, 0.2)
    out = encoder(out, lens)
    qout = encoder(qout, lens)

    # attention 需要考虑mask, 记得最后的start_idx & end_idx 也需要考虑mask
    attention = Attention(hidden_size=100, drop_rate=0.2)

    out_mask = torch.ones(4, 15).tril(0)
    qout_mask = torch.zeros(4, 10).tril(-1)

    global_out = attention(out, out_mask, qout, qout_mask)

    model_layer = ModelingLayer(hidden_size=100, drop_rate=0.2)
    out = model_layer(global_out)
    output_layer = Output(100)
    start_prob, end_prob = output_layer(global_out, out)

    print(111, start_prob.shape)
    print(111, end_prob.shape)


