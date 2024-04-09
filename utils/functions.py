import convpoint.knn.cpp.nearest_neighbors as nearest_neighbors
import numpy as np
import pyvista as pv
from sklearn.metrics import confusion_matrix

import utils.metrics as metrics

green_to_label = {255: 2, 192: 3, 129: 1, 64: 0}


def nearest_correspondence(pts_src, pts_dst, data_src, K=1):
    indices = nearest_neighbors.knn(pts_src, pts_dst, K, omp=True)
    if K == 1:
        indices = indices.ravel()
        data_dst = data_src[indices]
    else:
        # bug
        data_dst = data_src[indices].mean(1)
    return data_dst


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


def compute_metrics(labels, prediction, num_classes):
    cm = confusion_matrix(labels.flatten(), prediction.flatten(), labels=list(range(num_classes)))

    oa = metrics.stats_overall_accuracy(cm)
    aa = metrics.stats_accuracy_per_class(cm)[0]
    iou = metrics.stats_iou_per_class(cm)[0]
    return oa, aa, iou
