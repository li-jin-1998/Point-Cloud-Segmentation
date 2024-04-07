import h5py
import numpy as np


def read_h5_data(filename):
    try:
        with h5py.File(filename, 'r') as data:
            keys = list(data.keys())
            print(keys)

            print(type(data['data']))

            points = [data['data'][...].astype(np.float32)]
            labels = [data['label'][...].astype(np.int64)]
            point_nums = [data['data_num'][...].astype(np.int32)]
            labels_seg = [data['label_seg'][...].astype(np.int64)]

            print(points)
            print(np.concatenate(points, axis=0).shape)
            print(labels)
            print(point_nums)
            print(labels_seg[0].shape)

    except IOError as e:
        print(f"无法读取文件 {filename}: {e}")
    except Exception as e:
        print(f"处理文件 {filename} 时发生错误: {e}")


if __name__ == '__main__':
    filename_h5 = r'/mnt/algo_storage_server/PointCloudSeg/shapenet/shapenet_partseg/train_1.h5'
    read_h5_data(filename_h5)
