import numpy as np 
import matplotlib.pyplot as plt 

def plot_res(epochs, loss, train_acc, val_acc, test_acc):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, loss, marker='o', markersize=3, linestyle='-', color='b')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, train_acc, marker='o', markersize=3, linestyle='-', color='r', label='train_acc')
    ax2.plot(epochs, val_acc, marker='*', linestyle='-', color='g', label='valid_acc')
    ax2.plot([0, epochs[-1]], [test_acc, test_acc], linestyle='--', color='black', label='test_acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy / ,')
    ax2.set_title('Accuracy')
    ax2.legend()
    fig.tight_layout()

    plt.show()
