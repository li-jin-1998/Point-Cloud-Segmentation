import h5py
import numpy as np

filename_h5 = r'/mnt/algo_storage_server/PointCloudSeg/shapenet/shapenet_partseg/train_1.h5'

data = h5py.File(filename_h5, 'r')
keys = list(data.keys())
print(keys)

print(type(data['data']))

points = []
labels = []
point_nums = []
labels_seg = []

points.append(data['data'][...].astype(np.float32))
points.append(data['data'][...].astype(np.float32))
labels.append(data['label'][...].astype(np.int64))
point_nums.append(data['data_num'][...].astype(np.int32))
labels_seg.append(data['label_seg'][...].astype(np.int64))

print(points)
print(np.concatenate(points, axis=0).shape)
print(labels)
print(point_nums)
print(labels_seg[0].shape)
