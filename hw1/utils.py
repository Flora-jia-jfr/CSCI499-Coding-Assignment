import re
import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from transformers import BertTokenizer

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
        for inst, _ in episode:
            # strings already processed
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
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
    # print(int(np.max(padded_lens))) #57
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i for i, a in enumerate(actions)}
    targets_to_index = {t: i for i, t in enumerate(targets)}
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets


def encode_data(data, v2i, seq_len, a2i, t2i):
    n_lines = len(data)
    input = np.zeros((n_lines, seq_len), dtype=np.int32)
    out = np.zeros((n_lines, 2), dtype=np.int32)
    idx = 0
    n_unks = 0
    n_tks = 0
    for txt, output in data:
        action, target = output
        txt = preprocess_string(txt)
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


def plot_and_save(train_action_loss_record, train_target_loss_record, train_action_acc_record, train_target_acc_record,
                  train_epoch_record, valid_action_loss_record, valid_target_loss_record, valid_action_acc_record,
                  valid_target_acc_record, valid_epoch_record):
    plt.figure()
    # training action loss compared with valid action loss
    plt.subplot(2, 2, 1)
    plt.plot(train_epoch_record, train_action_loss_record, color='r', label="train action loss")
    plt.plot(valid_epoch_record, valid_action_loss_record, color='g', label="valid action loss")
    plt.xlabel("epoch")
    plt.ylabel("action loss")
    plt.legend(loc="best")
    plt.title('training v.s. valid action loss')
    # training target loss compared with valid target loss
    plt.subplot(2, 2, 2)
    plt.plot(train_epoch_record, train_target_loss_record, color='r', label="train target loss")
    plt.plot(valid_epoch_record, valid_target_loss_record, color='g', label="valid target loss")
    plt.xlabel("epoch")
    plt.ylabel("target loss")
    plt.legend(loc="best")
    plt.title('training v.s. valid target loss')
    # training action acc compared with valid action acc
    plt.subplot(2, 2, 3)
    plt.plot(train_epoch_record, train_action_acc_record, color='r', label="train action accuracy")
    plt.plot(valid_epoch_record, valid_action_acc_record, color='g', label="valid action accuracy")
    plt.xlabel("epoch")
    plt.ylabel("action accuracy")
    plt.legend(loc="best")
    plt.title('training v.s. valid action acc')
    # training target acc compared with valid target acc
    plt.subplot(2, 2, 4)
    plt.plot(train_epoch_record, train_target_acc_record, color='r', label="train target accuracy")
    plt.plot(valid_epoch_record, valid_target_acc_record, color='g', label="valid target accuracy")
    plt.xlabel("epoch")
    plt.ylabel("target accuracy")
    plt.legend(loc="best")
    plt.title('training v.s. valid target acc')
    plt.tight_layout()
    plt.savefig('train&valid_lost&accuracy.png')