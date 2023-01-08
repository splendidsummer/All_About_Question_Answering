from transformers import AutoModel
from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer, BertConfig, BertPreTrainedModel, \
AutoConfig, AutoTokenizer
from torch import nn
import torch.nn.functional as F
import config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelingLayer(nn.Module):
    def __init__(self, bert_hidden_size, lstm_hidden_size, drop_rate, num_layers=2):
        super(ModelingLayer, self).__init__()
        self.rnn = nn.LSTM(bert_hidden_size, lstm_hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=drop_rate)

    def forward(self, inputs):
        out, _ = self.rnn(inputs)
        return out


# BasicAtten
class BasicAttention(nn.Module):
    def __init__(self, bert_hidden_size, proj_size, drop_rate):
        super(BasicAttention, self).__init__()
        self.attention = nn.Linear(bert_hidden_size, 1)
        self.dropout = nn.Dropout(drop_rate)
        self.proj_ctx = nn.Linear(bert_hidden_size, proj_size)
        self.proj_ques = nn.Linear(bert_hidden_size, proj_size)

    def forward(self, context, context_masks, query, query_masks):

        # context.shape == query.shape == [bs, ctx_len/ques_len, bert_hidden_size]
        context = self.proj_ctx(self.dropout(context))
        query = self.proj_ques(self.dropout(query))
        batch_size, context_len, query_len = context.size(0), context.size(1), query.size(1)

        # context_.shape == query_.shape == [bs, ctx_len, ques_len, bert_hidden_size]
        context_ = context.unsqueeze(2).repeat(1, 1, query_len, 1)
        query_ = query.unsqueeze(1).repeat(1, context_len, 1, 1)
        # element-wise multiplication, so cq.shape == [bs, ctx_len, ques_len, bert_hidden_size]
        cq = torch.mul(context_, query_)

        # before view：sim_matrix.shape = [bs, ctx_len, ques_len, 1]
        # after view: sim_matrix.shape = [bs, ctx_len, ques_len]
        sim_matrix = self.attention(cq).view(-1, context_len, query_len)

        # alpha.shape == [bs, ctx_len, ques_len]
        alpha = F.softmax(sim_matrix, dim=-1)

        # content_masks.squeeze(2).shape = [bs, ctx_len, 1]
        context_masks = context_masks.unsqueeze(-1)
        # question_masks.squeeze(1).shape = [bs, 1, ques_len]
        question_masks = query_masks.unsqueeze(1)

        # alpha.shape == [bs, ctx_len, ques_len]
        alpha = alpha * context_masks * question_masks

        # [bs, context_len, query_len] * [bs, query_len, bert_hidden_size] ->
        # [bs, context_len, bert_hidden_size]
        a = torch.bmm(alpha, query)

        q2c_sim_mat, _ = torch.max(sim_matrix, dim=-1)
        # beta.shape = [bs, context_len]
        beta = F.softmax(q2c_sim_mat, dim=-1)
        #  [bs, ctx_len] * [bs, ctx_len]
        beta = beta * context_masks.squeeze()
        # beta.shape = [bs, 1, context_len]
        beta = beta.unsqueeze(1)

        # b.shape == [bs, 1, bert_hidden_size]
        b = torch.bmm(beta, context)
        # b.shape == [bs, ctx_len, bert_hidden_size]
        b = b.repeat(1, context_len, 1)
        c = torch.mul(context, a)
        d = torch.mul(context, b)

        global_hidden = torch.concat([context, a, c, d], dim=-1)

        return global_hidden


class SpanbertAttention(nn.Module):
    def __init__(self, bert_hidden_size, proj_size, drop_rate):
        super(SpanbertAttention, self).__init__()
        self.attention = BasicAttention(bert_hidden_size, proj_size, drop_rate)
        # self.dropout = nn.Dropout(drop_rate)

    def forward(self, inputs, ctx_mask, ques_mask, ctx_indices, ques_indices):
        batch_size, ctx_len, ques_len, hidden_size = \
            ctx_mask.size(0), ctx_mask.size(1), ques_mask.size(1), inputs.size(-1)

        ctx = torch.ones((batch_size, ctx_len, hidden_size)).to(device)
        ques = torch.ones((batch_size, ques_len, hidden_size)).to(device)

        for i in range(batch_size):
            ctx_idx = ctx_indices[i]
            ques_idx = ques_indices[i]
            ctx[i][: len(ctx_indices)] = inputs[ctx_idx]
            ques[i][: len(ques_idx)] = inputs[ques_idx]

        global_hiddens = self.attention(ctx, ctx_mask, ques, ques_mask)
        return global_hiddens


class SpanBert_BiDAF(BertPreTrainedModel):
    def __init__(self, bert_hidden_size, proj_size, lstm_hidden_size, num_layers, drop_rate):
        super(SpanBert_BiDAF, self).__init__(config=AutoConfig.from_pretrained(cfg.config_path))
        self.spanbert = AutoModel.from_pretrained(cfg.model_checkpoint)
        self.attention = SpanbertAttention(bert_hidden_size, proj_size, drop_rate)
        self.modeling_layer = ModelingLayer(bert_hidden_size, lstm_hidden_size, drop_rate, num_layers)
        self.dropout = nn.Dropout(drop_rate)
        self.fc_start = nn.Linear(proj_size * 4, 1)
        self.fc_end = nn.Linear(proj_size * 4, 1)

    def forward(self, input_ids, attention_mask, ctx_indices, ques_indices, ctx_mask, qs_mask):

        outputs = self.spanbert(input_ids, attention_mask=attention_mask)
        output = outputs[0]
        output = self.attention(output, ctx_mask, qs_mask, ctx_indices, ques_indices, ctx_mask, qs_mask)
        start_out = self.fc_start(output)
        end_out = self.fc_end(output)

        return start_out, end_out


if __name__ == '__main__':
    model = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
    print(model)
