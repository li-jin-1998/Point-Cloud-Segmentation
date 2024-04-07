import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

plot_save_path = r'/home/lij/PycharmProjects/Point-Cloud-Segmentation/log/plot/'

def loss_plot(args, train_loss, val_loss):
    num = args.epochs
    x = [i for i in range(1, num + 1)]
    os.makedirs(plot_save_path, exist_ok=True)

    save_loss = os.path.join(plot_save_path, f"{args.arch}_{args.epochs}_loss.png")
    plt.figure()
    plt.xlim([1, num + 1])
    plt.ylim([np.min(val_loss) / 2, np.max(val_loss) + 0.1])
    plt.plot(x, train_loss, 'r', label='train loss')
    plt.plot(x, val_loss, 'g', label='val loss')
    plt.legend()
    plt.savefig(save_loss, dpi=300)


def metrics_plot(arg, name, *value):
    num = arg.epochs
    names = name.split('&')
    metrics_value = value
    i = 0
    x = [i for i in range(1, num + 1)]

    os.makedirs(plot_save_path, exist_ok=True)
    save_metrics = plot_save_path + "{}_{}_{}.png".format(arg.arch, arg.epochs,
                                                          datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    plt.figure()
    for l in metrics_value:
        plt.plot(x, l, 'b', label=str(names[i]))
        # plt.scatter(x,l,label=str(l))
        i += 1
    plt.xlim([1, num + 1])
    plt.ylim([np.min(metrics_value) - 0.01, np.max(metrics_value) + 0.01])
    plt.legend()
    plt.savefig(save_metrics, dpi=300)


if __name__ == '__main__':
    from parse_args import parse_args

    args = parse_args()
    args.epochs = 5
    loss_plot(args, [0, 0.6, 0.8, 0.8, 0.92], [0, 0.3, 0.7, 0.86, 0.97])
    metrics_plot(args, "dice", [0.82, 0.86, 0.88, 0.92, 0.95])
