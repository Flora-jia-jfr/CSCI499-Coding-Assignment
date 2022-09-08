import json
import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch import nn
from transformers import BertTokenizer, BertModel
from bert_model import BertClassifier
from dataset_bert import BertDataset

from utils import (
    get_device,
    plot_and_save,
)

from torch.optim import Adam
from tqdm import tqdm


def train(model, train_data, val_data, learning_rate, epochs):

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataloader
    train, val = BertDataset(train_data), BertDataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=16)

    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad(): # do not update gradient
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


EPOCHS = 5
model = BertClassifier()
LR = 1e-6

data_file = open("lang_to_sem_data.json")
data = json.load(data_file)
train_instructions = []
valid_seen_instructions = []
data_file.close()
# contain 21023 groups, each group contain multiple tuples (instruction, [action, target])
# contain 820 groups, each group contain multiple tuples (instruction, [action, target])

# combine all the training data
for i in data["train"]:
    for group in i:
        txt, labels = group
        action, target = labels
        train_instructions.append([txt, target])
# print(len(train_instructions)) #
for i in data["valid_seen"]:
    for group in i:
        txt, labels = group
        action, target = labels
        valid_seen_instructions.append([txt, target])

train(model, train_instructions, valid_seen_instructions, LR, EPOCHS)
