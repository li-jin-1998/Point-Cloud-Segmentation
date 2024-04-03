import glob
import os
import random
import shutil
import time
import torch
import tqdm
import numpy as np
import pyvista as pv

from parse_args import parse_args, get_model
import convpoint.knn.cpp.nearest_neighbors as nearest_neighbors
from utils.process import save_ply_property

from sklearn.metrics import confusion_matrix
import utils.metrics as metrics

green_to_label = {255: 2, 192: 3, 129: 1, 64: 0}


def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    # start = time.time()
    indices = nearest_neighbors.knn(pts_src.copy(), pts_dest.copy(), K, omp=True)
    if K == 1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest


def read_ply(path, is_label=False):
    mesh = pv.read(path)
    points = mesh.points
    colors = mesh.active_scalars
    if is_label:
        labels = [green_to_label[i[1]] for i in colors]
        labels = np.array(labels).T

        return points, colors, labels, mesh.n_points
    else:
        return points, colors, mesh.n_points


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = get_model(args)
    weights_path = "save_weights/{}_{}_best_model.pth".format(args.arch, args.train_with_color)
    # load weights
    print(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    # torch.save(model.state_dict(), "save_weights/{}_predict_model.pth".format(args.arch))
    dst = '/mnt/algo_storage_server/PointCloudSeg/Dataset/test.txt'
    x = random.randint(1, 5000)
    predict_paths = [os.path.join(os.path.dirname(dst), 'data', line.strip()) for line in open(dst)][x:x + 1000]
    # print(predict_paths)

    result_path = './visualization'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)

    OA = []
    AA = []
    IOU = []

    start_time = time.time()
    model.eval()  # 进入验证模式
    with torch.no_grad():
        for path in tqdm.tqdm(predict_paths):
            labels = read_ply(path.replace('.ply', '_label.ply'), is_label=True)[2]
            # print(labels)
            ply_data = read_ply(path)
            pts = ply_data[0]
            choice = np.random.choice(ply_data[2], args.num_points, replace=True)

            pts = pts[choice]

            if args.train_with_color and len(ply_data[1]):
                cls = ply_data[1][choice]
                cls = cls.astype(np.float32)
                # cls = cls / 255 - 0.5
                cls = cls / 127.5 - 1
            else:
                cls = np.ones((pts.shape[0], 1))

            pts = torch.from_numpy(pts).float().to(device)
            cls = torch.from_numpy(cls).float().to(device)
            output = model(torch.unsqueeze(cls, dim=0), torch.unsqueeze(pts, dim=0)).squeeze(0)

            # cm = confusion_matrix(labels[choice].flatten(), output.argmax(1).flatten().cpu(), labels=list(range(args.num_classes)))
            #
            # oa = metrics.stats_overall_accuracy(cm)
            # aa = metrics.stats_accuracy_per_class(cm)[0]
            # iou = metrics.stats_iou_per_class(cm)[0]

            # interpolate to original points
            prediction = nearest_correspondance(pts.cpu().numpy(), ply_data[0], output)
            prediction = prediction.argmax(1).cpu().numpy()

            cm = confusion_matrix(labels.flatten(), prediction.flatten(), labels=list(range(args.num_classes)))

            oa = metrics.stats_overall_accuracy(cm)
            aa = metrics.stats_accuracy_per_class(cm)[0]
            iou = metrics.stats_iou_per_class(cm)[0]

            OA.append(oa)
            AA.append(aa)
            IOU.append(iou)

            save_ply_property(ply_data[0], prediction,
                              os.path.join(result_path, os.path.basename(path)))
            # print(os.path.join(result_path, os.path.basename(path)))

    print("oa:{:.2f} aa:{:.2f} miou:{:.2f}".format(np.mean(OA) * 100, np.mean(AA) * 100, np.mean(IOU) * 100))

    total_time = time.time() - start_time
    print("time {}s, fps {}".format(total_time, len(predict_paths) / total_time))


if __name__ == '__main__':
    main()
