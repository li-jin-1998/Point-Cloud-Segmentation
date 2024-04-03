import datetime
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.utils.data

from utils.train_and_eval import train_one_epoch, create_lr_scheduler, evaluate
from parse_args import parse_args, get_model, get_best_weight_path, get_latest_weight_path

from dataset import TeethDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Creating network...")
    model = get_model(args)
    model.cuda()
    print("parameters", count_parameters(model))

    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_dataset = TeethDataset(os.path.join(args.data_path, 'train_color.h5'), num_points=args.num_points,
                                 num_iter_per_shape=args.num_trees, train_with_color=args.train_with_color)
    val_dataset = TeethDataset(os.path.join(args.data_path, 'test_color.h5'), num_points=args.num_points,
                               num_iter_per_shape=args.num_trees, train_with_color=args.train_with_color)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adamax(params_to_optimize, lr=1e-3)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, last_epoch=-1, gamma=0.99, verbose=True)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, MILESTONES)

    # 用来保存训练以及验证过程中信息
    results_file = "log/{}_{}.txt".format(args.arch, datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        weights_path = "save_weights/{}_{}_best_model.pth".format(args.arch, args.train_with_color)
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch']
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        print(">" * 10, 'load best weight:', weights_path)

    best_miou = 0.
    best_oa = 0.
    best_aa = 0.
    best_epoch = 1
    train_losses = []
    val_losses = []
    mious = []
    lr = args.lr
    start_time = time.time()
    # create the log file
    logs = open(results_file, "a")
    for epoch_num in range(args.start_epoch, args.epochs + 1):
        print('-' * 20)
        print('Epoch {}/{} lr {:.6f}'.format(epoch_num, args.epochs, lr))
        print('-' * 20)

        train_loss, oa, aa, iou, lr = train_one_epoch(epoch_num, model, optimizer, train_loader, device,
                                                      args.num_classes, lr_scheduler=lr_scheduler, scaler=scaler)

        val_loss, val_oa, val_aa, val_miou = evaluate(epoch_num, model, val_loader, device, args.num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mious.append(val_miou)
        # write into txt
        # 记录每个epoch对应的train_loss、lr以及验证集各指标
        train_info = f"[epoch: {epoch_num}]\n" \
                     f"train_loss: {train_loss:.4f}\n" \
                     f"oa: {oa:.4f}\n" \
                     f"aa: {aa:.4f}\n" \
                     f"iou: {iou:.4f}\n"
        # write the logs
        logs.write(train_info + "\n\n")
        logs.flush()

        torch.save(model.state_dict(), get_latest_weight_path(args))
        if args.save_best is True:
            if best_miou <= val_miou:
                best_miou = val_miou
                best_oa = val_oa
                best_aa = val_aa
                best_epoch = epoch_num
                print("best epoch:{} oa:{:.2f} aa:{:.2f} miou:{:.2f}".format(best_epoch, best_oa * 100, best_aa * 100,
                                                                             best_miou * 100))
            else:
                print("best epoch:{} oa:{:.2f} aa:{:.2f} miou:{:.2f}".format(best_epoch, best_oa * 100, best_aa * 100,
                                                                             best_miou * 100))
                continue

        # save the model
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch_num,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, get_best_weight_path(args))

    best_info = f"[epoch: {best_epoch}]\n" \
                f"best_oa: {best_oa * 100:.2f}\n" \
                f"best_aa: {best_aa * 100:.2f}\n" \
                f"best_miou: {best_miou * 100:.2f}\n"
    logs.write(best_info)
    logs.flush()
    logs.close()

    from utils.plot import loss_plot, metrics_plot
    loss_plot(args, train_losses, val_losses)
    metrics_plot(args, "miou", mious)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    if not os.path.exists("./log"):
        os.mkdir("./log")
    train()
