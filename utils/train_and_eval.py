import sys

import numpy as np
import torch
import tqdm
from torch.nn.functional import cross_entropy

from utils.functions import compute_metrics


def criterion(output, target, loss_weight=None, num_classes: int = 3, label_smoothing: float = 0.1):
    if loss_weight is None:
        loss_weight = torch.ones(num_classes, device='cuda')
    # loss_weight = torch.as_tensor([1, 2, 2, 1], device="cuda")
    loss = cross_entropy(output.view(-1, num_classes), target.view(-1), weight=loss_weight,
                         label_smoothing=label_smoothing)
    return loss


def evaluate(epoch_num, model, data_loader, device, num_classes):
    model.eval()
    val_loss = []
    OA = []
    AA = []
    IOU = []
    with torch.no_grad():
        data_loader = tqdm.tqdm(data_loader, file=sys.stdout)
        for pts, fts, lbs, indices in data_loader:
            fts = fts.to(device)
            pts = pts.to(device)
            lbs = lbs.to(device)
            output = model(fts, pts)
            # print(output)

            loss = criterion(output, lbs, num_classes=num_classes)
            val_loss.append(loss.item())

            output_np = output.argmax(2).cpu().numpy()
            lbs_np = lbs.cpu().numpy()

            oa, aa, iou = compute_metrics(lbs_np.flatten(), output_np.flatten(), num_classes)

            OA.append(oa)
            AA.append(aa)
            IOU.append(iou)

            data_loader.set_postfix(OA=f"{np.mean(OA):.3f}", AA=f"{np.mean(AA):.3f}", IOU=f"{np.mean(IOU):.3f}")
            data_loader.desc = f"[val epoch {epoch_num}] loss: {np.mean(val_loss):.4f}"

    return np.mean(val_loss), np.mean(OA), np.mean(AA), np.mean(IOU)


def train_one_epoch(epoch_num, model, optimizer, data_loader, device, num_classes,
                    lr_scheduler, scaler=None):
    model.train()

    train_loss = []
    OA = []
    AA = []
    IOU = []
    data_loader = tqdm.tqdm(data_loader, file=sys.stdout)
    for pts, cls, lbs, indices in data_loader:
        cls = cls.to(device)
        pts = pts.to(device)
        lbs = lbs.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(cls, pts)
            loss = criterion(output, lbs, num_classes=num_classes)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        train_loss.append(loss.item())

        output_np = output.argmax(2).cpu().numpy()
        lbs_np = lbs.cpu().numpy()

        oa, aa, iou = compute_metrics(lbs_np.flatten(), output_np.flatten(), num_classes)

        OA.append(oa)
        AA.append(aa)
        IOU.append(iou)

        data_loader.set_postfix(OA=f"{np.mean(OA):.3f}", AA=f"{np.mean(AA):.3f}", IOU=f"{np.mean(IOU):.3f}")
        data_loader.desc = f"[train epoch {epoch_num}] loss: {np.mean(train_loss):.4f}"
    # lr_scheduler.step()

    return np.mean(train_loss), np.mean(OA), np.mean(AA), np.mean(IOU), lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.95

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
