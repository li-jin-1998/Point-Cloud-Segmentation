import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(args):
    print('**************************')
    print('model:{}\nepochs:{}\nbatch size:{}\nnum points:{}'
          .format(args.arch, args.epochs, args.batch_size, args.num_points))
    print('**************************')
    if args.arch == "SegSmall":
        from networks.network_seg import SegSmall
        model = SegSmall(input_channels=3 if args.train_with_color else 1, output_channels=args.num_classes).to(device)
    if args.arch == "SegBig":
        from networks.network_seg import SegBig
        model = SegBig(input_channels=3 if args.train_with_color else 1, output_channels=args.num_classes).to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch training")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='SegBig',
                        help='SegSmall/SegBig')
    parser.add_argument("--data_path", default='/mnt/algo_storage_server/PointCloudSeg/Dataset', help="root")
    # exclude background
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--num_points", default=5000, type=int)
    parser.add_argument("--train_with_color", default=0, type=bool)
    parser.add_argument("--num_trees", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--resume', default=0, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    model = get_model(args)
