# IMPLEMENT YOUR MODEL CLASS HERE
import torch
from torch import nn


class ActionTargetPredict(torch.nn.Module):
    def __init__(self, device, vocab_size, input_len, n_actions, n_targets, embedding_dim, num_hiddens, num_layers):
        super(ActionTargetPredict, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.n_actions = n_actions
        self.n_targets = n_targets

        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM
        self.model = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=num_hiddens,
            num_layers=num_layers
        )

        # dropout layer (optional)
        # self.dropout = nn.Dropout()

        # linear layer
        # thinking of two ways:
        # 1) output (action, target) -- 8 * 80 = 640 possible outputs => this may keep in the information of the
        # relationship between the action and the target -- at least not very likely to output (pick up, drawer)
        # 2) separately output action and target using two nn.Linear
        self.hidden2action = torch.nn.Linear(num_hiddens, n_actions)
        self.hidden2target = torch.nn.Linear(num_hiddens, n_targets)

    def forward(self, input):
        # print("input: ", input.shape)
        embeds = self.embedding(input.permute(1, 0))
        # print("embeds: ", embeds.shape)
        outputs, _ = self.model(embeds)
        # print("outputs: ", outputs.shape)
        out_action = self.hidden2action(outputs[-1]).squeeze(1)
        # print("out_action_shape: ", out_action.shape)
        out_target = self.hidden2target(outputs[-1]).squeeze(1)
        # print("out_target_shape: ", out_target.shape)

        return out_action, out_target
