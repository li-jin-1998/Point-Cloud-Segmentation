import numpy as np


def random_sample(num_points, num_choices):
    # 如果选取的数目大于可选数目，使用抽样方法
    if num_choices > num_points:
        # 随机抽样，确保选取的数字不会重复
        indices = np.random.choice(num_points, size=num_points, replace=False)
        # indices = np.arange(0, num_points)
        # 重复抽样直到满足选取数目
        indices = np.concatenate([indices, np.random.choice(num_points, size=num_choices - num_points, replace=True)])
    else:
        # 如果可选数目大于等于选取的数目，直接进行随机选择
        indices = np.random.choice(num_points, size=num_choices, replace=False)
    return indices


class VisualizationPoints:
    def __init__(self, points_list, colors_list, point_size=4, opacity=1.0):
        self.pl = None
        self.meshes = points_list
        self.colors = colors_list
        self.point_size = point_size
        self.opacity = opacity
        self.num_meshes = len(points_list)
        self.init_plotter()

    def init_plotter(self):
        self.pl = pv.Plotter(shape=(1, self.num_meshes))
        self.pl.set_background([0.9, 0.9, 0.9])
        self.pl.add_key_event("s", self.save_screenshot)

    def display(self):
        self.pl.clear()
        for i in range(self.num_meshes):
            self.pl.subplot(0, i)

            assert self.meshes[i].shape[0] == self.colors[i].shape[0]

            self.pl.add_title(str(self.meshes[i].shape[0]), font_size=20, font='times')
            self.pl.show_axes()
            self.pl.add_points(self.meshes[i], scalars=self.colors[i], point_size=self.point_size, style="points",
                               rgb=True, opacity=self.opacity)
            self.pl.view_xy()

        self.pl.show()

    def save_screenshot(self):
        self.pl.screenshot('./screenshot.png')
        print('save screenshot')


if __name__ == '__main__':
    import pyvista as pv

    file_path = './test2.ply'
    mesh = pv.read(file_path)
    points = mesh.points
    colors = mesh.active_scalars

    choice = random_sample(len(mesh.points), 5000)
    # choice = np.random.choice(len(mesh.points), 4000, replace=True)
    print(np.unique(choice).shape)

    sampled_points = points[choice]
    sampled_colors = colors[choice]

    print(points.shape, sampled_points.shape, sampled_colors.shape)
    # pl = pv.Plotter(shape=(1, 2))
    #
    # pl.subplot(0, 0)
    # pl.add_points(points, scalars=colors, point_size=5, style="points", rgb=True)
    # pl.add_title(str(points.shape[0]), font_size=20, font='times')
    # pl.view_xy()
    #
    # pl.subplot(0, 1)
    # pl.add_points(sampled_points, scalars=sampled_colors, point_size=5, style="points", rgb=True)
    # pl.add_title(str(sampled_points.shape[0]), font_size=20, font='times')
    # pl.view_xy()
    # pl.show()

    vis = VisualizationPoints(points_list=[points, sampled_points], colors_list=[colors, sampled_colors])
    vis.display()
