import torch
from transformers import BertTokenizer, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)


class FCModel(torch.nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc = torch.nn.Linear(in_features=768, out_features=1)

    def forward(self, input):
        score = self.fc(input)
        result = torch.sigmoid(score)
        return result


# from transformers import BertTokenizer, BertModel
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
import torch
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, device, n_actions, n_targets, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.device = device
        self.n_actions = n_actions
        self.n_targets = n_targets

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        # BERT base has 768 hidden units
        self.hidden2action = torch.nn.Linear(768, n_actions)
        self.hidden2target = torch.nn.Linear(768, n_targets)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        out_action = self.hidden2action(dropout_output)
        out_target = self.hidden2target(dropout_output)
        return out_action, out_target

