import json
import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from model import ActionTargetPredict
from utils import (
    get_device,
    encode_data,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    plot_and_save,
)


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


def setup_model(args, device, vocab_size, input_len, n_actions, n_targets):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    embedding_dim, num_hiddens, num_layers = 100, 128, 2
    # device, vocab_size, input_len, n_actions, n_targets, embedding_dim, num_hiddens, num_layers
    model = ActionTargetPredict(device, vocab_size, input_len, n_actions, n_targets, embedding_dim, num_hiddens,
                                num_layers).to(device)
    return model


def setup_optimizer(args, model, device):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    learning_rate = 0.0001
    action_criterion = torch.nn.CrossEntropyLoss().to(device)
    target_criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
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

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs)

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
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
        model.train()

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
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
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
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

    # build modelnn
    # model = setup_model(args, maps, device)
    model = setup_model(args, device, vocab_size, input_len, n_actions, n_targets)
    print(model)
    # (embedding): Embedding(1000, 100, padding_idx=0)
    # (model): LSTM(100, 128, num_layers=2)
    # (dropout): Dropout(p=0.5, inplace=False)
    # (hidden2action): Linear(in_features=128, out_features=8, bias=True)
    # (hidden2target): Linear(in_features=128, out_features=80, bias=True)


    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model, device)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )
        torch.save(model, args.model_output_dir+"entire_model.pt")


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

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()
    main(args)
