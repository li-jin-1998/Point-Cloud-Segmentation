import datetime
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.utils.data

from utils.train_and_eval import evaluate
from parse_args import parse_args, get_model, get_best_weight_path

from dataset import TeethDataset


def val():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Creating network...")
    model = get_model(args)
    weights_path = get_best_weight_path(args)
    print(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    start_time = time.time()

    batch_size = args.batch_size * 2
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_dataset = TeethDataset(os.path.join(args.data_path, 'test_color.h5'), num_points=args.num_points,
                               num_iter_per_shape=args.num_trees, train_with_color=args.train_with_color)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)

    val_loss, oa, aa, iou = evaluate(0, model, val_loader, device, args.num_classes)

    print("val loss:{:.4f}".format(val_loss))
    print("val oa:{:.2f}".format(oa * 100))
    print("val aa:{:.2f}".format(aa * 100))
    print("val iou:{:.2f}".format(iou * 100))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("test time {}".format(total_time_str))


if __name__ == '__main__':
    val()
