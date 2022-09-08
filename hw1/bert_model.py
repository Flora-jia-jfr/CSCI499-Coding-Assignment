import torch
from transformers import BertTokenizer, BertModel

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert_model = BertModel.from_pretrained("bert-base-uncased")
# bert_model.to(device)

# class FCModel(torch.nn.Module):
#     def __init__(self, device, n_actions, n_targets):
#         super(FCModel, self).__init__()
#         self.feature2action = torch.nn.Linear(in_features=768, out_features=n_actions)
#         self.feature2target = torch.nn.Linear(in_features=768, out_features=n_targets)
#
#     def forward(self, input):
#         action_score = self.feature2action(input)
#         target_score = self.feature2target(input)
#         action = torch.softmax(action_score)
#         target = torch.softmax(target_score)
#         return action, target

import torch
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.n_actions = 8
        self.n_targets = 80

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        # BERT base has 768 hidden units
        # self.hidden2action = torch.nn.Linear(768, n_actions)
        self.hidden2target = torch.nn.Linear(768, 80)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        # out_action = self.hidden2action(dropout_output)
        out_target = self.hidden2target(dropout_output)
        final_out_target = self.relu(out_target)
        return final_out_target

