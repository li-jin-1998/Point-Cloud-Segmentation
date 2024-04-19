import os
import random
import time

import numpy as np
import torch
import tqdm

import path_manager
from parse_args import parse_args, get_model, get_best_weight_path
from random_sampling import random_sample
from utils.functions import read_ply, nearest_correspondence, compute_metrics


def predict():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = get_model(args)
    weights_path = get_best_weight_path(args)
    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    # torch.save(model.state_dict(), "save_weights/{}_predict_model.pth".format(args.arch))
    dst = '/mnt/algo_storage_server/PointCloudSeg/Dataset/test.txt'
    x = random.randint(1, 5000)
    predict_paths = [os.path.join(os.path.dirname(dst),
                                  'data', line.strip()) for line in open(dst)][::10]  # [x:x + 1000]

    result_path = './visualization'
    path_manager.safe_create_directory(result_path)

    OA = []
    AA = []
    IOU = []

    start_time = time.time()
    model.eval()  # 进入验证模式
    with torch.no_grad():
        for path in tqdm.tqdm(predict_paths):
            labels = read_ply(path.replace('.ply', '_label.ply'), is_label=True)[2]
            ply_data = read_ply(path)
            pts = ply_data[0]
            # choice = np.random.choice(ply_data[2], args.num_points, replace=True)
            choice = random_sample(ply_data[2], args.num_points)
            # print(ply_data[2], np.unique(choice).shape[0])
            pts = pts[choice]

            if args.use_color and len(ply_data[1]):
                cls = ply_data[1][choice]
                cls = cls.astype(np.float32)
                # cls = cls / 255 - 0.5
                cls = cls / 127.5 - 1
            else:
                cls = np.ones((pts.shape[0], 1))

            pts = torch.from_numpy(pts).float().to(device)
            cls = torch.from_numpy(cls).float().to(device)
            output = model(torch.unsqueeze(cls, dim=0), torch.unsqueeze(pts, dim=0)).squeeze(0)

            # interpolate to original points
            # prediction = ball_nearest_neighbors(pts.cpu().numpy(), ply_data[0], output)
            prediction = nearest_correspondence(pts.cpu().numpy(), ply_data[0], output)
            prediction = prediction.argmax(1).cpu().numpy()

            oa, aa, iou = compute_metrics(labels, prediction, args.num_classes)

            OA.append(oa)
            AA.append(aa)
            IOU.append(iou)

            # save_ply_property(ply_data[0], prediction, os.path.join(result_path, os.path.basename(path)))

    print("oa:{:.2f} aa:{:.2f} miou:{:.2f}".format(np.mean(OA) * 100, np.mean(AA) * 100, np.mean(IOU) * 100))

    total_time = time.time() - start_time
    print("time {}s, fps {}".format(total_time, len(predict_paths) / total_time))


if __name__ == '__main__':
    predict()
