import json

import numpy as np
import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from bert_model import FCModel

from utils import (
    get_device,
    plot_and_save,
)

def encode_data(data):
    n_lines = len(data)
    out = np.zeros((n_lines, 2), dtype=np.int32)
    idx = 0
    n_unks = 0
    n_tks = 0
    for txt, output in data:
        action, target = output
        input[idx][0] = v2i["<start>"]
        jdx = 1
        for word in txt.split():
            if len(word) > 0:
                if word in v2i:
                    input[idx][jdx] = v2i[word]
                else:
                    input[idx][jdx] = v2i["<unk>"]
                    n_unks += 1
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    break
        input[idx][jdx] = v2i["<end>"]
        out[idx][0] = a2i[action]
        out[idx][1] = t2i[target]
        idx += 1
    return input, out


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validation set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #

    # 1. read in all data
    data_file = open(args.in_data_fn)
    data = json.load(data_file)
    train_instructions = []
    valid_seen_instructions = []
    data_file.close()
    # contain 21023 groups, each group contain multiple tuples (instruction, [action, target])
    # contain 820 groups, each group contain multiple tuples (instruction, [action, target])

    # combine all the training data
    for i in data["train"]:
        for group in i:
            train_instructions.append(group)
    # print(len(train_instructions)) #
    for i in data["valid_seen"]:
        for group in i:
            valid_seen_instructions.append(group)

    for input, label in train_instructions:


    # 2. build the tokenizer_table
    vocab_to_index, index_to_vocab, instruction_len = build_tokenizer_table(data["train"])
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(data["train"])
    # print(len(actions_to_index)) # 8
    # print(len(targets_to_index)) # 80

    # 3. encode the training and validation set inputs/outputs

    train_np_input, train_np_output = encode_data(train_instructions, vocab_to_index, instruction_len,
                                                                   actions_to_index, targets_to_index)
    train_dataset = TensorDataset(torch.from_numpy(train_np_input), torch.from_numpy(train_np_output))
    valid_np_input, valid_np_output = encode_data(valid_seen_instructions, vocab_to_index, instruction_len,
                                                                   actions_to_index, targets_to_index)
    valid_dataset = TensorDataset(torch.from_numpy(valid_np_input), torch.from_numpy(valid_np_output))

    # 4. create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)
    vocab_size = len(vocab_to_index)
    n_actions = len(actions_to_index)
    n_targets = len(targets_to_index)
    return train_loader, val_loader, vocab_size, instruction_len, n_actions, n_targets


def setup_model(args, device, n_actions, n_targets):
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # bert_model = BertModel.from_pretrained("bert-base-uncased")
    # bert_model.to(device)
    FC_model = FCModel(device, n_actions, n_targets)
    FC_model = FC_model.to(device)
    return FC_model


def setup_optimizer(args, device, FC_model, bert_model):
    # for fully connected layer
    FC_optimizer = torch.optim.Adam(FC_model.parameters(), lr=0.001)
    # bert only needs small changes
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=0.00001)
    action_criterion = torch.nn.BCELoss()
    target_criterion = torch.nn.BCELoss()
    return action_criterion, target_criterion, FC_optimizer, bert_optimizer


def train_epoch(
    args,
    FC_model,
    bert_model,
    loader,
    fc_optimizer,
    bert_optimizer,
    action_criterion,
    target_criterion,
    device,
    tokenizer,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        print(inputs)
        encoding = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
        bert_output = bert_model(**encoding.to(device))
        pooler_output = bert_output.pooler_output
        actions_out, targets_out = FC_model(pooler_output)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        # actions_out, targets_out = FC_model(inputs)

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())

        loss = action_loss + target_loss


        # step optimizer and compute gradients during training
        if training:
            fc_optimizer.zero_grad()
            bert_optimizer.zero_grad()
            loss.backward()
            fc_optimizer.step()
            bert_optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss/len(action_labels), epoch_target_loss/len(action_labels), action_acc, target_acc


def validate(
    args, fc_model, bert_model, loader, fc_optimizer, bert_optimizer, action_criterion, target_criterion, device, tokenizer
):
    # set model to eval mode
    fc_model.eval()
    bert_model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            fc_model,
            bert_model,
            loader,
            fc_optimizer,
            bert_optimizer,
            action_criterion,
            target_criterion,
            device,
            tokenizer,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, fc_model, bert_model, loaders, fc_optimizer, bert_optimizer, action_criterion, target_criterion, device, tokenizer):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    # model.train()

    train_action_loss_record = []
    train_target_loss_record = []
    train_action_acc_record = []
    train_target_acc_record = []
    train_epoch_record = []
    valid_action_loss_record = []
    valid_target_loss_record = []
    valid_action_acc_record = []
    valid_target_acc_record = []
    valid_epoch_record = []

    train_epoch_count = 1
    valid_epoch_count = args.val_every


    for epoch in tqdm.tqdm(range(args.num_epochs)):
        # set back to model.train() after evaluation
        bert_model.train()
        fc_model.train()

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            fc_model,
            bert_model,
            loaders["train"],
            fc_optimizer,
            bert_optimizer,
            action_criterion,
            target_criterion,
            device,
            tokenizer
        )

        train_action_loss_record.append(train_action_loss)
        train_action_acc_record.append(train_action_acc)
        train_target_loss_record.append(train_target_loss)
        train_target_acc_record.append(train_target_acc)
        train_epoch_record.append(train_epoch_count)
        train_epoch_count += 1

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy, but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                fc_model,
                bert_model,
                loaders["val"],
                fc_optimizer,
                bert_optimizer,
                action_criterion,
                target_criterion,
                device,
                tokenizer
            )

            valid_action_loss_record.append(val_action_loss)
            valid_action_acc_record.append(val_action_acc)
            valid_target_loss_record.append(val_target_loss)
            valid_target_acc_record.append(val_target_acc)
            valid_epoch_record.append(valid_epoch_count)
            valid_epoch_count += args.val_every

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target acc: {val_target_acc}"
            )

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #

    # train_action_loss_record = []
    # train_target_loss_record = []
    # train_action_acc_record = []
    # train_target_acc_record = []
    # train_epoch_record = []
    # valid_action_loss_record = []
    # valid_target_loss_record = []
    # valid_action_acc_record = []
    # valid_target_acc_record = []
    # valid_epoch_record = []
    plot_and_save(train_action_loss_record, train_target_loss_record, train_action_acc_record, train_target_acc_record,
                  train_epoch_record, valid_action_loss_record, valid_target_loss_record, valid_action_acc_record,
                  valid_target_acc_record, valid_epoch_record)


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    # what are the maps for???
    # train_loader, val_loader, maps = setup_dataloader(args)
    train_loader, val_loader, vocab_size, input_len, n_actions, n_targets = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.to(device)

    # build modelnn
    # model = setup_model(args, maps, device)

    FC_model = setup_model(args, device, n_actions, n_targets)
    print(bert_model)
    print(FC_model)
    # (embedding): Embedding(1000, 100, padding_idx=0)
    # (model): LSTM(100, 128, num_layers=2)
    # (dropout): Dropout(p=0.5, inplace=False)
    # (hidden2action): Linear(in_features=128, out_features=8, bias=True)
    # (hidden2target): Linear(in_features=128, out_features=80, bias=True)


    # get optimizer and loss functions

    action_criterion, target_criterion, fc_optimizer, bert_optimizer = setup_optimizer(args, device, FC_model, bert_model)

    # if args.eval:
    #     val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
    #         args,
    #         model,
    #         loaders["val"],
    #         optimizer,
    #         action_criterion,
    #         target_criterion,
    #         device,
    #     )
    # else:
    train(
        args, FC_model, bert_model, loaders, fc_optimizer, bert_optimizer, action_criterion, target_criterion, device, tokenizer
    )
    # torch.save(model, args.model_output_dir+"entire_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", type=int, default=5, help="number of epochs between every eval loop"
    )

    args = parser.parse_args()
    main(args)
