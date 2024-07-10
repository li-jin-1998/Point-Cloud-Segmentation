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


if __name__ == '__main__':
    import pyvista as pv

    file_path = '../test2.ply'
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
    from vis_tools.visualization import VisualizationPoints

    vis = VisualizationPoints(points_list=[points, sampled_points], colors_list=[colors, sampled_colors])
    vis.display()
