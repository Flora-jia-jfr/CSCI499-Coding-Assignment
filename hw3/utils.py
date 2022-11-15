import json
import re
import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train:
        padded_len = 2  # start/end
        for inst, _ in episode:
            inst = preprocess_string(inst)
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
        padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
              : vocab_size - 4
              ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        # int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
        200
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i+3 for i, a in enumerate(actions)}
    targets_to_index = {t: i+3 for i, t in enumerate(targets)}
    actions_to_index["<start>"] = 0
    actions_to_index["<stop>"] = 1
    actions_to_index["<pad>"] = 2
    targets_to_index["<start>"] = 0
    targets_to_index["<stop>"] = 1
    targets_to_index["<pad>"] = 2

    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets


def prefix_match(predicted_labels, gt_labels):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1 
    # print("--------------IN PREFIX MATCH--------------")
    batch_size = gt_labels.shape[0]
    seq_length = gt_labels.shape[1]
    # print("batch_size: ", batch_size)
    # print("seq_length: ", seq_length)
    # print("gt_labels: ", gt_labels.shape)
    # print("predicted_labels: ", predicted_labels.shape)
    match = 0.0
    sum = torch.count_nonzero(gt_labels)
    # print("sum of the gt_labels: ", sum)
    for i in range(batch_size):
        for j in range(seq_length):
            if predicted_labels[i][j] == gt_labels[i][j]:
                match += 1
            else:
                break
    # pm = (1.0 / seq_length) * i
    # print("match: ", match)
    pm = float(match/sum)
    # print("pm: ", pm)
    # torch.set_printoptions(threshold=10000)
    # print("predicted_labels: ", predicted_labels)
    # print("target_labels: ", gt_labels)
    # raise RuntimeError("stop and let me check")
    return pm

def exact_match(predicted_labels, gt_labels):
    # print("--------------IN EXACT MATCH--------------")
    batch_size = gt_labels.shape[0]
    seq_length = gt_labels.shape[1]
    # print("batch_size: ", batch_size)
    # print("seq_length: ", seq_length)
    # print("gt_labels: ", gt_labels.shape)
    # print("predicted_labels: ", predicted_labels.shape)
    match = 0.0
    sum = torch.count_nonzero(gt_labels)
    for i in range(batch_size):
        for j in range(seq_length):
            if predicted_labels[i][j] == gt_labels[i][j] and predicted_labels[i][j] != 1:
                match += 1
    # pm = (1.0 / seq_length) * i
    # print("match: ", match)
    exact_acc = float(match/sum)
    # print("exact_acc: ", exact_acc)
    return exact_acc

def read_data_from_file(file_path):
    """
        read in data from json file into two arrays for output (train, valid_seen)
    """
    # TODO: change back, currently for debug
    max_episode_len = 0  # train already has longer instructions for one episode
    with open(file_path) as data_file:
        data = json.load(data_file)
        for i in data["train"]:
        # for i in data["train"][0: 6]:
            max_episode_len = max(len(i), max_episode_len)
    return data["train"], data["valid_seen"], max_episode_len + 2  # add one for <stop>,<stop>
    # return data["train"][0: 6], data["train"][0: 6], max_episode_len + 2
    # add one for <stop>,<stop> and <start>,<start>


def encode_data(episodes, vocab_to_index, seq_len, actions_to_index, targets_to_index, max_episode_len):
    """
        encode data into inputs and outputs for the model
    """
    # print("seq_len: ", seq_len)  # 605
    # print("max_episode_len: ", max_episode_len)  # 605
    n_episodes = len(episodes)
    input = np.zeros((n_episodes, seq_len), dtype=np.int32)
    output = np.ones((n_episodes, max_episode_len, 2), dtype=np.int32)
    episode_index = 0
    for episode in episodes:
        word_index = 0
        label_index = 0
        input[episode_index][word_index] = vocab_to_index["<start>"]
        output[episode_index][label_index][0] = actions_to_index["<start>"]
        output[episode_index][label_index][1] = targets_to_index["<start>"]
        word_index += 1
        label_index += 1
        for txt, out in episode:
            action, target = out
            txt = preprocess_string(txt)
            for word in txt.split():
                if word_index == seq_len - 1:
                    break
                if len(word) > 0:
                    if word in vocab_to_index:
                        input[episode_index][word_index] = vocab_to_index[word]
                    else:
                        input[episode_index][word_index] = vocab_to_index["<unk>"]
                    word_index += 1
            output[episode_index][label_index][0] = actions_to_index[action]
            output[episode_index][label_index][1] = targets_to_index[target]
            label_index += 1
        input[episode_index][word_index] = vocab_to_index["<end>"]
        output[episode_index][label_index][0] = actions_to_index["<stop>"]
        output[episode_index][label_index][1] = targets_to_index["<stop>"]
        episode_index += 1
    return input, output

def save_and_plot(train_action_loss_record, train_target_loss_record, val_action_loss_record, val_target_loss_record,
                  val_action_prefix_acc_record, val_target_prefix_acc_record, val_action_exact_acc_record,
                  val_target_exact_acc_record, train_epoch_record, valid_epoch_record):

    print("train_action_loss_record: ", train_action_loss_record)
    print("train_target_loss_record: ", train_target_loss_record)
    print("val_action_loss_record: ", val_action_loss_record)
    print("val_target_loss_record: ", val_target_loss_record)
    print("val_action_prefix_acc_record: ", val_action_prefix_acc_record)
    print("val_target_prefix_acc_record: ", val_target_prefix_acc_record)
    print("val_action_exact_acc_record: ", val_action_exact_acc_record)
    print("val_target_exact_acc_record: ", val_target_exact_acc_record)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(train_epoch_record, train_action_loss_record, color='r', label="train action loss")
    plt.plot(valid_epoch_record, val_action_loss_record, color='g', label="valid action loss")
    plt.xlabel("epoch")
    plt.ylabel("action loss")
    plt.legend(loc="best")
    plt.title('training v.s. valid action loss')

    plt.subplot(2, 2, 2)
    plt.plot(train_epoch_record, train_target_loss_record, color='r', label="train target loss")
    plt.plot(valid_epoch_record, val_target_loss_record, color='g', label="valid target loss")
    plt.xlabel("epoch")
    plt.ylabel("target loss")
    plt.legend(loc="best")
    plt.title('training v.s. valid target loss')

    plt.subplot(2, 2, 3)
    plt.plot(valid_epoch_record, val_action_prefix_acc_record, color='r', label="val action prefix acc")
    plt.plot(valid_epoch_record, val_action_exact_acc_record, color='g', label="val action match acc")
    plt.xlabel("epoch")
    plt.ylabel("action acc")
    plt.legend(loc="best")
    plt.title('val action acc')

    plt.subplot(2, 2, 4)
    plt.plot(valid_epoch_record, val_target_prefix_acc_record, color='r', label="val target prefix acc")
    plt.plot(valid_epoch_record, val_target_exact_acc_record, color='g', label="val target match acc")
    plt.xlabel("epoch")
    plt.ylabel("target acc")
    plt.legend(loc="best")
    plt.title('val target acc')

    plt.tight_layout()
    plt.savefig('train&valid_lost&accuracy.png')