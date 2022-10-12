import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#
# arr = np.array([[1,2,3],[4,5,6]], dtype=int)
# np.save("data.npy", arr)
# brr = np.load("data.npy")
# print("arr: ", arr)
# print("brr: ", brr)
# if os.path.exists("encoded_sentences.csv") and os.path.exists("lens.csv"):
#     print("Reading stored encoded sentences...")
#     encoded_sentences = pd.read_csv('encoded_sentences.csv').to_numpy()
#     lens = pd.read_csv('lens.csv').to_numpy()
#     print(encoded_sentences.shape, lens.shape)
#     print(encoded_sentences[-5:])
#     print(lens[-5:])

train_loss_record = [1,2,3,4,5]
train_acc_record = [5,4,3,2,1]
val_loss_record = [2,5,8,10,13,66]
val_acc_record = [5,6,7,6,9,66]
#
# def draw_single_plot(train_loss_record, num_epochs):
#     plt.figure()
#     # training action loss compared with valid action loss
#     plt.plot(range(num_epochs), train_loss_record, color='r', label="train loss")
#     plt.xlabel("epoch")
#     plt.ylabel("train loss")
#     plt.title('train loss with CBOW')
#     plt.savefig('CBOW_train_loss.png')

def plot_and_save(train_loss_record, train_acc_record, val_loss_record, val_acc_record, num_epochs):
    # train loss figure
    plt.figure()
    plt.plot(range(num_epochs), train_loss_record, color='r', label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title('train loss with CBOW')
    plt.savefig('plots/CBOW_train_loss.png')
    plt.show()

    # train accuracy figure
    plt.figure()
    plt.plot(range(num_epochs), train_acc_record, color='r', label="train acc")
    plt.xlabel("epoch")
    plt.ylabel("train accuracy")
    plt.title('train loss with CBOW')
    plt.savefig('plots/CBOW_train_acc.png')
    plt.show()

    # val loss figure
    plt.figure()
    plt.plot(range(0, 30, 5), val_loss_record, color='g', label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("val loss")
    plt.title('train loss with CBOW')
    plt.savefig('plots/CBOW_val_loss.png')
    plt.show()

    # val accuracy figure
    plt.figure()
    plt.plot(range(0, 30, 5), val_acc_record, color='r', label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("val accuracy")
    plt.title('val loss with CBOW')
    plt.savefig('plots/CBOW_val_acc.png')
    plt.show()


plot_and_save(train_loss_record, train_acc_record, val_loss_record, val_acc_record,5)


a = "what"
print(f"{a}azhe")