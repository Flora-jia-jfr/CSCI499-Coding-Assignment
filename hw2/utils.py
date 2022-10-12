import json
import gensim
import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

def read_analogies(analogies_fn):
    with open(analogies_fn, "r") as f:
        pairs = json.load(f)
    return pairs


def save_word2vec_format(fname, model, i2v):
    print("Saving word vectors to file...")  # DEBUG
    with gensim.utils.open(fname, "wb") as fout:
        fout.write(
            gensim.utils.to_utf8("%d %d\n" % (model.vocab_size, model.embedding_dim))
        )
        # store in sorted order: most frequent words at the top
        for index in tqdm.tqdm(range(len(i2v))):
            word = i2v[index]
            row = model.embedding.weight.data[index]
            fout.write(
                gensim.utils.to_utf8(
                    "%s %s\n" % (word, " ".join("%f" % val for val in row))
                )
            )


def get_device(force_cpu, status=True):
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device

def create_train_val_splits(encoded_sentences, lens, prop_train=0.7):
    print("create_train_val_splits")
    val_idxs = np.random.choice(list(range(len(encoded_sentences))),
                                size=int(len(encoded_sentences) * (1 - prop_train) + 0.5), replace=False)
    val_sentences = []
    val_lens = []
    train_sentences = []
    train_lens = []
    train_sentences.extend([encoded_sentences[idx] for idx in range(len(encoded_sentences)) if idx not in val_idxs])
    train_lens.extend([lens[idx] for idx in range(len(lens)) if idx not in val_idxs])
    val_sentences.extend([encoded_sentences[idx] for idx in range(len(encoded_sentences)) if idx in val_idxs])
    val_lens.extend([lens[idx] for idx in range(len(lens)) if idx in val_idxs])
    return train_sentences, train_lens, val_sentences, val_lens


def sentences2pair(encoded_sentences, lens, context_length):
    """
        input:
            encoded_sentences: train_sentences or val_sentences
            lens: train_lens or val_lens
        output:
            input_context: np.array containing list of length 2*k+1
            output_word: np.array containing word
    """
    input_context = []
    output_word = []

    for index, sentence in enumerate(encoded_sentences):
        curr_len = lens[index][0]  # current sentence length

        if (context_length * 2 + 1) > curr_len:
            continue
        for word_index in range(context_length, curr_len - context_length):  # all possible words
            chosen_context = []
            for i in range(word_index-context_length, word_index+context_length+1):
                if i == word_index:
                    output_word.append(sentence[i])
                else:
                    chosen_context.append(sentence[i])
            input_context.append(chosen_context)

    # print("input_context length: ", len(input_context), input_context[-5:-1])
    # print("output_word length: ", len(output_word), output_word[-5:-1])
    return np.array(input_context), np.array(output_word)


def plot_and_save(train_loss_record, train_acc_record, val_loss_record, val_acc_record, num_epochs, plot_folder, val_every):
    # train loss figure
    plt.figure()
    plt.plot(range(0, num_epochs), train_loss_record, color='r', label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title('train loss with CBOW')
    plt.savefig(f"{plot_folder}/CBOW_train_loss.png")
    plt.show()

    # train accuracy figure
    plt.figure()
    plt.plot(range(0, num_epochs), train_acc_record, color='r', label="train acc")
    plt.xlabel("epoch")
    plt.ylabel("train accuracy")
    plt.title('train accuracy with CBOW')
    plt.savefig(f"{plot_folder}/CBOW_train_acc.png")
    plt.show()

    # val loss figure
    plt.figure()
    plt.plot(range(0, num_epochs, val_every), val_loss_record, color='g', label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("val loss")
    plt.title('val loss with CBOW')
    plt.savefig(f"{plot_folder}/CBOW_val_loss.png")
    plt.show()

    # val accuracy figure
    plt.figure()
    plt.plot(range(0, num_epochs, val_every), val_acc_record, color='r', label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("val accuracy")
    plt.title('val accuracy with CBOW')
    plt.savefig(f"{plot_folder}/CBOW_val_acc.png")
    plt.show()
