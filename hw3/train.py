import tqdm
import torch
import argparse

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

from model import (
    Encoder,
    Decoder,
    EncoderDecoder, BERTEncoder,
)
from utils import (
    get_device,
    preprocess_string,
    read_data_from_file,
    build_tokenizer_table,
    build_output_tables,
    prefix_match, encode_data, exact_match, save_and_plot
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
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    # 1. read json file into arrays
    train_episode, valid_seen_episode, max_episode_len = read_data_from_file(args.in_data_fn)
    # print(len(train_episode), len(valid_seen_episode))
    # 2. build tokenizer table
    vocab_to_index, index_to_vocab, instruction_len = build_tokenizer_table(train_episode)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_episode)

    # 3. encode training and validation set input/outputs
    train_input, train_output = encode_data(train_episode, vocab_to_index, instruction_len, actions_to_index,
                                            targets_to_index, max_episode_len)
    valid_input, valid_output = encode_data(valid_seen_episode, vocab_to_index, instruction_len, actions_to_index,
                                            targets_to_index, max_episode_len)
    train_dataset = TensorDataset(torch.from_numpy(train_input), torch.from_numpy(train_output))
    valid_dataset = TensorDataset(torch.from_numpy(valid_input), torch.from_numpy(valid_output))

    # 4. create dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)
    vocab_size = len(vocab_to_index)
    n_actions = len(actions_to_index)
    n_targets = len(targets_to_index)

    return train_loader, val_loader, vocab_size, instruction_len, n_actions, n_targets


def setup_model(args, device, vocab_size, instruction_len, num_actions, num_targets):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    embedding_dim, label_embedding_dim, hidden_dim, num_layers = 128, 64, 100, 1
    encoder = Encoder(device, vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    decoder = Decoder(device, label_embedding_dim, hidden_dim, num_layers, num_actions, num_targets).to(device)
    if args.model == "vanilla":
        model = EncoderDecoder(device, encoder, decoder, num_actions, num_targets, hidden_dim, attention=False).to(device)
    elif args.model == "attention":
        model = EncoderDecoder(device, encoder, decoder, num_actions, num_targets, hidden_dim, attention=True).to(device)
    elif args.model == "transformer":
        embedding_dim, label_embedding_dim, hidden_dim, num_layers = 128, 64, 768, 1
        bert_encoder = BERTEncoder(device, embedding_dim, hidden_dim).to(device)
        decoder = Decoder(device, label_embedding_dim, hidden_dim, num_layers, num_actions, num_targets).to(device)
        model = EncoderDecoder(device, bert_encoder, decoder, num_actions, num_targets, hidden_dim, attention=True).to(device)
    return model


def setup_optimizer(args, device, model, num_actions, num_targets):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    # TODO: change back for normal training
    learning_rate = 1e-3
    action_weight = torch.softmax(torch.tensor([2] + [1]*(num_actions-1)).float(), dim=-1)
    # print("action_weight: ", action_weight.shape, action_weight)
    # print(action_weight.sum())
    target_weight = torch.softmax(torch.tensor([2] + [1] * (num_targets-1)).float(), dim=-1)
    # print("action_weight: ", target_weight.shape, target_weight)
    # print(target_weight.sum())
    action_criterion = torch.nn.CrossEntropyLoss(weight=action_weight, ignore_index=1).to(device)
    target_criterion = torch.nn.CrossEntropyLoss(weight=target_weight, ignore_index=1).to(device)
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
        teacher_forcing=True
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_action_loss = 0.0
    epoch_target_loss = 0.0
    epoch_action_prefix_acc = 0.0
    epoch_target_prefix_acc = 0.0
    epoch_action_exact_acc = 0.0
    epoch_target_exact_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        action_outputs, target_outputs = model(inputs, labels, teacher_forcing=teacher_forcing)

        predicted_action_outputs = torch.argmax(action_outputs, dim=2).to(device)
        predicted_target_outputs = torch.argmax(target_outputs, dim=2).to(device)
        # print("predicted_action_outputs: ", predicted_action_outputs.shape)
        # print("predicted_target_outputs: ", predicted_target_outputs.shape)

        # TODO: debug
        # print("check loss:")
        # print("action_outputs.squeeze(): ", action_outputs.squeeze().shape)
        # print(labels.shape)
        # print(labels[:, :, 0].shape)
        # print("action_outputs: ", action_outputs.shape)
        action_loss = action_criterion(action_outputs.permute(0, 2, 1).to(device), labels[:, :, 0].long())
        target_loss = target_criterion(target_outputs.permute(0, 2, 1).to(device), labels[:, :, 1].long())
        # print("action_loss: ", action_loss)
        # print("target_loss: ", target_loss)
        # action accuracy increase but target accuracy decrease during epochs
        # (so my guess is that target is not being focused enough)
        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        if not training:
            action_prefix_em = prefix_match(predicted_action_outputs, labels[:, :, 0])
            target_prefix_em = prefix_match(predicted_target_outputs, labels[:, :, 0])
            action_em = exact_match(predicted_action_outputs, labels[:, :, 0])
            target_em = exact_match(predicted_target_outputs, labels[:, :, 0])


        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()
        if not training:
            epoch_action_prefix_acc += action_prefix_em
            epoch_target_prefix_acc += target_prefix_em
            epoch_action_exact_acc += action_em
            epoch_target_exact_acc += target_em

        print("action_loss: ", action_loss, "| target_loss: ", target_loss)
        # print("action_prefix_em: ", action_prefix_em)
        # print("target_prefix_em: ", target_prefix_em)
        # print("action_em: ", action_em)
        # print("target_em: ", target_em)

        # raise RuntimeError("Stop and Check")

    epoch_action_loss /= len(loader)
    epoch_target_loss /= len(loader)
    if not training:
        epoch_action_prefix_acc /= len(loader)
        epoch_target_prefix_acc /= len(loader)
        epoch_action_exact_acc /= len(loader)
        epoch_target_exact_acc /= len(loader)

    return epoch_action_loss, epoch_target_loss, epoch_action_prefix_acc, epoch_target_prefix_acc, \
           epoch_action_exact_acc, epoch_target_exact_acc


def validate(args, model, loader, optimizer, action_criterion, target_criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_action_loss, val_target_loss, val_action_prefix_acc, val_target_prefix_acc, val_action_exact_acc,\
            val_target_exact_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
            teacher_forcing=False
        )

    return val_action_loss, val_target_loss, val_action_prefix_acc, val_target_prefix_acc, val_action_exact_acc,\
            val_target_exact_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    train_action_loss_record = []
    train_target_loss_record = []

    val_action_loss_record = []
    val_target_loss_record = []
    val_action_prefix_acc_record = []
    val_target_prefix_acc_record = []
    val_action_exact_acc_record = []
    val_target_exact_acc_record = []

    train_epoch_record = []
    valid_epoch_record = []
    train_epoch_count = 1
    valid_epoch_count = args.val_every


    for epoch in tqdm.tqdm(range(args.num_epochs)):
        # set back to model.train() after evaluation
        model.train()
        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_action_loss, train_target_loss, train_action_prefix_acc, train_target_prefix_acc, train_action_exact_acc, \
            train_target_exact_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
            teacher_forcing=True
        )
        train_action_loss_record.append(train_action_loss)
        train_target_loss_record.append(train_target_loss)
        train_epoch_record.append(train_epoch_count)
        train_epoch_count += 1

        # some logging
        print(f"train action loss: {train_action_loss} | train target loss: {train_target_loss}")
        # print(f"train action prefix accuracy: {train_action_prefix_acc}, train target prefix accuracy: {train_target_prefix_acc}")
        # print(f"train action exact accuracy: {train_action_exact_acc}, train target exact accuracy: {train_target_exact_acc}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_prefix_acc, val_target_prefix_acc, val_action_exact_acc, \
            val_target_exact_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            val_action_loss_record.append(val_action_loss)
            val_target_loss_record.append(val_target_loss)
            val_action_prefix_acc_record.append(val_action_prefix_acc)
            val_target_prefix_acc_record.append(val_target_prefix_acc)
            val_action_exact_acc_record.append(val_action_exact_acc)
            val_target_exact_acc_record.append(val_target_exact_acc)
            valid_epoch_record.append(valid_epoch_count)
            valid_epoch_count += args.val_every

            print(f"val action loss : {val_action_loss} | val target loss: {val_target_loss}")
            print(f"val action prefix accuracy: {val_action_prefix_acc}, val target prefix accuracy: {val_target_prefix_acc}")
            print(f"val action exact accuracy: {val_action_exact_acc}, val target exact accuracy: {val_target_exact_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #
    save_and_plot(train_action_loss_record, train_target_loss_record, val_action_loss_record, val_target_loss_record,
                  val_action_prefix_acc_record, val_target_prefix_acc_record, val_action_exact_acc_record,
                  val_target_exact_acc_record, train_epoch_record, valid_epoch_record)

def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, vocab_size, instruction_len, n_actions, n_targets = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device, vocab_size, instruction_len, n_actions, n_targets)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, device, model,
                                                                    num_actions=n_actions, num_targets=n_targets)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, action_criterion, target_criterion, device)


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
    parser.add_argument("--num_epochs", default=30, type=int, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, type=int, help="number of epochs between every eval loop"
    )
    parser.add_argument("--model", default="vanilla", type=str, help="choose from vanilla, attention, or transformer")

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
