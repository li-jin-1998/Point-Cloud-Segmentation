import numpy as np
import h5py

from utils import data_utils

color_map = {2: (102, 255, 205), 3: (30, 192, 100), 1: (255, 129, 80), 0: (64, 64, 192)}


def save_ply_property(points, property, filename):
    point_num = points.shape[0]
    colors = np.full(points.shape, 0.5)
    # print(point_num, colors.shape, property.shape)
    for point_idx in range(point_num):
        if property[point_idx] == -1:
            colors[point_idx] = np.array([0, 0, 0])
        else:
            colors[point_idx] = color_map[property[point_idx]]
    # print(colors)
    data_utils.save_ply(points, filename, colors / 255)


def load_seg_list(file_path):
    points = []
    colors = []
    labels = []
    point_nums = []
    if file_path is not dict:
        file_path = [file_path]
        # print(file_path)
    for file in file_path:
        data = h5py.File(file, 'r')
        points.append(data['data'][...].astype(np.float32))
        colors.append(data['color'][...].astype(np.float32))
        labels.append(data['label'][...].astype(np.int64))
        point_nums.append(data['data_num'][...].astype(np.int32))

    return (np.concatenate(points, axis=0),
            np.concatenate(colors, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(point_nums, axis=0))


def load_seg(file_path):
    data = h5py.File(file_path, 'r')
    points = data['data'][...].astype(np.float32)
    colors = data['color'][...].astype(np.float32) if 'color' in data.keys() else []
    labels = data['label'][...].astype(np.int64)
    points_num = data['data_num'][...].astype(np.int32)

    return points, colors, labels, points_num
