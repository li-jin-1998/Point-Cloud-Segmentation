import glob
import os
import shutil

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import utils.metrics as metrics
from predict import read_ply
from utils.process import save_ply_property


def compute_metrics(labels, prediction, num_classes):
    cm = confusion_matrix(labels.flatten(), prediction.flatten(), labels=list(range(num_classes)))

    oa = metrics.stats_overall_accuracy(cm)
    aa = metrics.stats_accuracy_per_class(cm)[0]
    iou = metrics.stats_iou_per_class(cm)[0]
    return oa, aa, iou


def compute_metrics_from_ply(gt_path, pred_path):
    pred = read_ply(pred_path, is_label=True)
    gt = read_ply(gt_path, is_label=True)
    diff = pred[2] == gt[2]
    save_ply_property(pred[0], diff,
                      os.path.join(result_path, os.path.basename(gt_path)))
    # oa, aa, iou = compute_metrics(pred[2], gt[2], 4)
    # print(oa, aa, iou)


def diff_ply(gt_paths, pred_paths):
    for gt_path, pred_path in tqdm(zip(gt_paths, pred_paths)):
        # print(gt_path, pred_path)
        compute_metrics_from_ply(gt_path, pred_path)


if __name__ == '__main__':
    result_path = './diff'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    src = '/mnt/algo_storage_server/PointCloudSeg/Dataset/data/'
    pred_paths = glob.glob(r'./visualization/*')
    print(len(pred_paths))
    preds = [p for p in pred_paths if 'ply' in p and 'label' not in p]
    gts = []
    for pred_path in preds:
        src_path = os.path.join(src, os.path.basename(pred_path))
        gt_path = src_path.replace('.ply', '_label.ply')
        gts.append(gt_path)
    diff_ply(gts, preds)
