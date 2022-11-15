import json

import torch
import numpy as np
from transformers import BertTokenizer

from utils import build_output_tables

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# 1. read in all data
data_file = open("lang_to_sem_data.json")
data = json.load(data_file)
train_episodes = []
valid_episodes = []
data_file.close()
# contain 21023 groups, each group contain multiple tuples (instruction, [action, target])
# contain 820 groups, each group contain multiple tuples (instruction, [action, target])

# combine all the training data
for i in data["train"]:
    curr_episode_instruction = ""
    curr_episode_label = []
    for instruction, label in i:
        curr_episode_instruction += instruction
        curr_episode_label.append(label)  # [[a, t], [a, t]]
    group = [curr_episode_instruction, curr_episode_label]
    train_episodes.append(group)
# print(len(train_instructions)) #
for i in data["valid_seen"]:
    curr_episode_instruction = ""
    curr_episode_label = []
    for instruction, label in i:
        curr_episode_instruction += instruction
        curr_episode_label.append(label)
    group = [curr_episode_instruction, curr_episode_label]
    valid_episodes.append(group)

actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(data["train"])


class BertDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        labels = []
        texts = []
        for txt, target in df:
            # action, target = l
            # labels.append(actions_to_index[action], targets_to_index[target])
            labels.append(targets_to_index[target])
            texts.append(tokenizer(txt, padding='max_length', max_length = 512, truncation=True, return_tensors="pt"))

        self.labels = labels
        self.texts = texts

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y