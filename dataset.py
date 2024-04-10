import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

from random_sampling import random_sample
from utils.process import load_seg

torch.manual_seed(3407)


class PointCloudDataset(Dataset):

    def __init__(self, path, num_points, num_iter_per_shape=1, train_with_color=True):
        # Prepare inputs
        print('{}-Preparing datasets: {}'.format(datetime.datetime.now(), path))
        points, colors, labels, points_num = load_seg(path)
        print("Done", points.shape)
        self.points = points
        self.points_num = points_num
        self.labels = labels
        self.colors = colors

        self.num_points = num_points
        self.train_with_color = train_with_color
        self.num_iter_per_shape = num_iter_per_shape

    def __getitem__(self, index):
        index = index // self.num_iter_per_shape

        npts = self.points_num[index]
        pts = self.points[index, :npts]

        # choice = np.random.choice(npts, self.num_points, replace=True)
        choice = random_sample(npts, self.num_points)

        pts = pts[choice]
        lbs = self.labels[index][choice]

        # separate between features and points
        if self.train_with_color and len(self.colors):
            cls = self.colors[index][choice]
            cls = cls.astype(np.float32)
            # cls = cls / 255 - 0.5
            cls = cls / 127.5 - 1
        else:
            cls = np.ones((pts.shape[0], 1))

        pts = torch.from_numpy(pts).float()
        cls = torch.from_numpy(cls).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, cls, lbs, index

    def __len__(self):
        return self.points.shape[0] * self.num_iter_per_shape


if __name__ == '__main__':
    # print(torch.cuda.is_bf16_supported())
    import os

    from parse_args import parse_args

    args = parse_args()
    test_dataset = PointCloudDataset(os.path.join(args.data_path, 'test_color.h5'), num_points=args.num_points,
                                     train_with_color=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=8)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for p, c, l, indices in test_loader:
        c = c.to(device)
        p = p.to(device)
        l = l.to(device)
        # print(cls, lbs)
        print(p.shape, c.shape, l.shape)
