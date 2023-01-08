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


class BasicAttention(nn.Module):
    def __init__(self, bert_hidden_size, drop_rate):
        super(BasicAttention, self).__init__()
        self.attention = nn.Linear(bert_hidden_size, 1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, context, context_masks, query, query_masks):

        # context.shape == query.shape == [bs, ctx_len/ques_len, bert_hidden_size]
        context = self.dropout(context)
        query = self.dropout(query)
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
        #  [bs, 1, ctx_len] * [bs, ctx_len]
        beta = beta * context_masks.squeeze()
        beta = beta.unsqueeze(1)

        # beta.shape = [bs, 1, context_len]
        b = torch.bmm(beta, context).repeat(1, context_len, 1)
        c = torch.mul(context, a)
        d = torch.mul(context, b)
        global_hidden = torch.concat([context, a, c, d], dim=-1)

        return global_hidden


class SpanbertAttention(nn.Module):
    def __init__(self, hidden_size, drop_rate):
        super(SpanbertAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, inputs, ctx_mask, ques_mask, ctx_indices, ques_indices):
        batch_size, ctx_len, ques_len, hidden_size = \
            ctx_mask.size(0), ctx_mask.size(1), ques_mask.size(1), inputs.size(-1)

        ctx = torch.ones((batch_size, ctx_len, hidden_size)).to(device)
        ques = torch.ones((batch_size, ques_len, hidden_size)).to(device)

        for i in range(batch_size):
            ctx_idx = ctx_indices[i]
            ques_idx = ques_indices[i]
            ctx[i][: len(ctx_indices)] = inputs[ctx_idx]
            ques[i][: len()] = inputs[ques_idx]

        return None


class SpanBert_BiDAF(BertPreTrainedModel):
    def __init__(self, bert_hidden_size, attention_hidden_size, lstm_hidden_size,
                 num_layers, drop_rate):
        super(SpanBert_BiDAF, self).__init__(config=AutoConfig.from_pretrained(cfg.config_path))
        self.spanbert = AutoModel.from_pretrained(cfg.model_checkpoint)
        self.attention = SpanbertAttention(attention_hidden_size, drop_rate)
        self.modeling_layer = ModelingLayer(bert_hidden_size, lstm_hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(lstm_hidden_size*2, 1)
        self.fc = nn.Linear(bert_hidden_size, 1)

    def forward(self, input_ids, attention_mask, start_positions, end_positions, context_indices,
                ques_indices, ctx_mask, qs_mask):

        outputs = self.spanbert(input_ids, attention_mask=attention_mask)
        output = outputs[0]
        output = self.attention(output)
        #
        out = self.fc(output)
        out = F.softmax(self.dropout(out))

        print(output.shape)
        print(out.shape)

        return out

class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.spanbert = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.lstm_embedding_size,  # 1024
            hidden_size=config.hidden_size // 2,  # 1024
            batch_first=True,
            num_layers=2,
            dropout=config.lstm_dropout_prob,  # 0.5
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs


if __name__ == '__main__':
    model = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
    print(model)
