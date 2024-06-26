import glob
import os
import shutil

from tqdm import tqdm

from predict import read_ply
from utils.process import save_ply_property


def compute_metrics_from_ply(gt_path, pred_path):
    pred = read_ply(pred_path, is_label=True)
    gt = read_ply(gt_path, is_label=True)
    diff = pred[2] == gt[2]
    save_ply_property(pred[0], diff,
                      os.path.join(result_path, os.path.basename(gt_path)))
    # oa, aa, iou = compute_metrics(pred[2], gt[2], 4)
    # print(oa, aa, iou)


if __name__ == '__main__':
    result_path = './diff'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)

    src = '/mnt/algo_storage_server/PointCloudSeg/Dataset/data/'
    pred_paths = glob.glob(r'./visualization/*')
    print(len(pred_paths))

    preds = [p for p in pred_paths if 'ply' in p and 'label' not in p]
    for pred_path in tqdm(preds):
        src_path = os.path.join(src, os.path.basename(pred_path))
        gt_path = src_path.replace('.ply', '_label.ply')
        compute_metrics_from_ply(gt_path, pred_path)
