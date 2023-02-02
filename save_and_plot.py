import sys
import json
import os
import collections
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
matplotlib.use('TKAgg')


def load_epoch_loss(new_data, path):
    if os.path.getsize(path) == 0:
        old_data = {}
    else:
        with open(path, 'r') as f:
            old_data = json.load(f)
    old_data.update(new_data)
    # with open(path, 'r') as f:
    #     file = f.read()
    #     if len(file) <= 0:
    #         old_data = json.load(f)
    #     else:
    #         old_data = {}
    #     old_data.update(new_data)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(old_data, f, indent=4, sort_keys=True)


def draw_loss_figures(train_epoch_loss_list, val_epoch_loss_list, iter_list, save_path):
    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    ax = [[ax1, ax2], [ax3, ax4], [ax5, ax6]]

    train_label_list = ["train_loss", "pos_loss", "neg_loss"]
    val_label_list = ["val_loss", "pos_loss", "neg_loss"]
    marker_list = ["kx-", "r.-", "b.-"]
    max_loss = float("-inf")
    for i, loss in enumerate(train_epoch_loss_list):
        ax[i][0].plot(iter_list, loss, marker_list[i], label=train_label_list[i])
        if max_loss < max(loss):
            max_loss = max(loss)

    for i, loss in enumerate(val_epoch_loss_list):
        ax[i][1].plot(iter_list, loss, marker_list[i], label=val_label_list[i])
        if max_loss < max(loss):
            max_loss = max(loss)

    ax[0][0].set_title("Training Loss")
    ax[0][1].set_title("Validation Loss")

    ax[0][0].set_ylabel("loss")
    ax[1][0].set_ylabel("loss")
    ax[2][0].set_ylabel("loss")

    ax[2][0].set_xlabel("epoch")
    ax[2][1].set_xlabel("epoch")


    x_major_locator = MultipleLocator(1)

    ax[0][0].xaxis.set_major_locator(x_major_locator)
    ax[0][1].xaxis.set_major_locator(x_major_locator)
    ax[1][0].xaxis.set_major_locator(x_major_locator)
    ax[1][1].xaxis.set_major_locator(x_major_locator)
    ax[2][0].xaxis.set_major_locator(x_major_locator)
    ax[2][1].xaxis.set_major_locator(x_major_locator)

    for axi in ax:
        for axj in axi:
            axj.set_ylim([0, max_loss])

    figure_path = os.path.join(save_path, "train_val_loss_epoch.jpg")
    fig.savefig(figure_path)
    plt.close('all')


def draw_epoch_loss(path):
    with open(path, encoding='utf-8') as f:
        json_data = json.load(f)
        json_data = {int(k): v for k, v in json_data.items()}
        od = collections.OrderedDict(sorted(json_data.items()))
        epoch_loss = []
        pos_epoch_loss = []
        neg_epoch_loss = []
        epoch = []
        pos_energy_per_epoch = []
        neg_energy_per_epoch = []
        for i in range(len(od)):
            epoch_loss.append(od[i + 1]["epoch_loss"])
            pos_epoch_loss.append(od[i + 1]["pos_epoch_loss"])
            neg_epoch_loss.append(od[i + 1]["neg_epoch_loss"])
            epoch.append(i + 1)
            pos_energy_per_epoch.append(od[i + 1]["pos_energy"])
            neg_energy_per_epoch.append(od[i + 1]["neg_energy"])
    return [epoch_loss, pos_epoch_loss, neg_epoch_loss], epoch, pos_energy_per_epoch, neg_energy_per_epoch


def draw_hist(pos_energy_per_epoch, neg_energy_per_epoch, i, path):
    bins = np.linspace(min(pos_energy_per_epoch + neg_energy_per_epoch),
                       max(pos_energy_per_epoch + neg_energy_per_epoch), 30)
    fig, ax = plt.subplots()
    ax.hist(pos_energy_per_epoch, bins, alpha=0.5, label='real data')
    ax.hist(neg_energy_per_epoch, bins, alpha=0.5, label='fake data')
    ax.set_xlabel(r'classifier score ($-E(\theta)$)')
    ax.set_ylabel("count")
    ax.legend()
    figure_name = ("energy_score_in_epoch_{}.jpg".format(i))
    figure_path = os.path.join(path, figure_name)
    fig.savefig(figure_path)
    plt.close()


if __name__ == "__main__":
    train_path = './output/result/LJSpeech\\train_loss.json'
    val_path = './output/result/LJSpeech\\valid_loss.json'
    output_path = './output/result/LJSpeech'

    # load train and valid output from json file
    train_epoch, iter_list, train_pos_energy, train_neg_energy = draw_epoch_loss(train_path)
    val_epoch, iter_list, val_pos_energy, val_neg_energy = draw_epoch_loss(val_path)

    # draw figure of epoch loss for train and valid output
    draw_loss_figures(train_epoch, val_epoch, iter_list, output_path)

    # draw figure of energy distribution for each epoch
    # sys.setrecursionlimit(1500)
    for i in iter_list:
        draw_hist(train_pos_energy[i-1], train_neg_energy[i-1], i, output_path + '/train_energy/')
        draw_hist(val_pos_energy[i-1], val_neg_energy[i-1], i, output_path + '/val_energy/')
