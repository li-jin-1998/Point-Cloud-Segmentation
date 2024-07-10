import argparse

import torch

from networks.network_seg import SegSmall, SegBig


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    return device


def get_best_weight_path(args):
    weights_path = "save_weights/{}_{}_{}_best_model.pth".format(args.arch, args.use_color, args.num_points)
    print("load weight: ", weights_path)
    return weights_path


def get_latest_weight_path(args):
    weights_path = "save_weights/{}_{}_{}_latest_model.pth".format(args.arch, args.use_color, args.num_points)
    # print(weights_path)
    return weights_path


def get_model(args):
    print('**************************')
    print('model:{}\nepochs:{}\nbatch size:{}\nnum points:{}'
          .format(args.arch, args.epochs, args.batch_size, args.num_points))
    print('**************************')
    if args.arch == "SegSmall":
        return SegSmall(input_channels=3 if args.use_color else 1, output_channels=args.num_classes).to(
            get_device())
    if args.arch == "SegBig":
        return SegBig(input_channels=3 if args.use_color else 1, output_channels=args.num_classes).to(
            get_device())


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch training")

    # Model architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='SegBig',
                        help='Segmentation model architecture (SegSmall/SegBig)')

    # Data path with environment variable as default
    parser.add_argument("--data_path", default='/mnt/algo-storage-server/Projects/PointCloudSeg/Dataset',
                        help="root path to the dataset")

    # Training parameters
    parser.add_argument("--num_classes", default=4, type=int, help="number of classes excluding background")
    parser.add_argument("--num_points", default=2000, type=int, help="number of points to train with")
    parser.add_argument("--use_color", default=1, type=int, help="whether to train with color")
    parser.add_argument("--num_trees", default=1, type=int, help="number of trees in the random forest")

    # Training configuration
    parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size for training")
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--resume', default=0, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')

    # Model saving configuration
    parser.add_argument('--save_best', default=True, type=bool, help='only save the best dice weights')

    # Mixed precision training
    parser.add_argument("--amp", default=False, type=bool, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    model = get_model(parse_args())
