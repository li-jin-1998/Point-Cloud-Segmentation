import datetime
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.utils.data

from utils.train_and_eval import evaluate
from parse_args import parse_args, get_model, get_best_weight_path, get_device

from dataset import PointCloudDataset


def val():
    args = parse_args()
    device = get_device()

    print("Creating network...")
    model = get_model(args)
    weights_path = get_best_weight_path(args)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    start_time = time.time()

    batch_size = args.batch_size * 2
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_dataset = PointCloudDataset(os.path.join(args.data_path, 'test_color.h5'), num_points=args.num_points,
                                    num_iter_per_shape=args.num_trees, use_color=args.use_color)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)

    val_loss, val_oa, val_oa, val_miou = evaluate(0, model, val_loader, device, args.num_classes)

    val_info = f"val_loss: {val_loss:.4f}\n" \
               f"val_oa: {val_oa:.4f}\n" \
               f"val_aa: {val_oa:.4f}\n" \
               f"val_miou: {val_miou:.4f}\n"
    print(val_info)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("test time {}".format(total_time_str))


if __name__ == '__main__':
    val()
