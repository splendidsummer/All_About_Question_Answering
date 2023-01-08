from transformers import AutoModel
import config as cfg
from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer, BertConfig, BertPreTrainedModel, \
AutoConfig, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class SpanbertDataset(Dataset):

    def __init__(self, datasets, subset='train'):
        super(SpanbertDataset, self).__init__()
        self.input_ids = datasets[subset]['input_ids']
        self.attention_mask = datasets[subset]['attention_mask']
        self.start_positions = datasets[subset]['start_positions']
        self.end_positions = datasets[subset]['end_positions']
        self.context_indices = datasets[subset]['context_indices']
        self.ques_indices = datasets[subset]['ques_indices']

    def __len__(self):
        return len(self.start_positions)

    def __getitem__(self, item):
        input_ids = self.input_ids[item]
        attention_mask = self.attention_mask[item]
        start_positions = self.start_positions[item]
        end_positions = self.end_positions[item]
        context_indices = self.context_indices[item]
        ques_indices = self.ques_indices[item]

        return input_ids, attention_mask, start_positions, end_positions, context_indices, ques_indices

    def batch_data_pro(self, batch_datas):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        context_indices = [i for (_, _, _, _, i, _) in batch_datas]
        ques_indices = [i for (_, _, _, _, _, i) in batch_datas]
        ctx_mask = torch.tensor(self._get_mask(context_indices), dtype=torch.long, device=device)
        qs_mask = torch.tensor(self._get_mask(ques_indices), dtype=torch.long, device=device)
        input_ids = torch.tensor([i for (i, _, _, _, _, _) in batch_datas], dtype=torch.long, device=device)
        attention_mask = torch.tensor([i for (_, i, _, _, _, _) in batch_datas], dtype=torch.long, device=device)
        start_positions = torch.tensor([i for (_, _, i, _, _, _) in batch_datas], dtype=torch.long, device=device)
        end_positions = torch.tensor([i for (_, _, _, i, _, _) in batch_datas], dtype=torch.long, device=device)

        return input_ids, attention_mask, start_positions, end_positions, context_indices, ques_indices, ctx_mask, qs_mask

    @staticmethod
    def _get_mask(sent_ids):
        sent_lens = [len(sent) for sent in sent_ids]
        max_len = max(sent_lens)
        masks = torch.zeros((len(sent_ids), max_len))
        for i, length in enumerate(sent_lens):
            masks[i, : length] = 1

        return masks















