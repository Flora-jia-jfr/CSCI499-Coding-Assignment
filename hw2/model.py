# IMPLEMENT YOUR MODEL CLASS HERE
import numpy as np
import torch
from torch import nn


class CBOW(torch.nn.Module):
    def __init__(self, device, vocab_size, embedding_dim, context_length):
        super(CBOW, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.vocab_size = vocab_size

        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        # fully connected layer
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.Linear(embedding_dim, 2 * embedding_dim),
            torch.nn.Linear(2 * embedding_dim, vocab_size),
        )

    def forward(self, input):
        # input of 2k words
        # print("input: ", input.shape)
        embeds = torch.sum(self.embedding(input), dim=1)
        # embeds = self.embedding(input.permute(1, 0))
        # embedding_sum = torch.sum(embeds, dim=0)
        # print("embeds: ", embeds.shape)
        predict_word = self.fcn(embeds)
        # print("outputs: ", predict_word.shape)

        return predict_word