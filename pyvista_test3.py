import os
import glob
import random

import pyvista as pv


def display_multi_meshes(meshes: list, titles=None, point_size=3, opacity=0.9):
    # from pyvista.plotting import themes
    # my_theme = themes.DocumentTheme()
    # my_theme.color = 'red'
    # my_theme.lighting = True
    # my_theme.show_edges = True
    # my_theme.edge_color = 'red'
    # my_theme.background = 'gray'
    # pv.set_plot_theme(my_theme)
    num = len(meshes)

    pl = pv.Plotter(shape=(1, num))
    pl.set_background([0.9, 0.9, 0.9])

    for i in range(num):
        pl.subplot(0, i)
        if titles is not None:
            pl.add_title(titles[i], font_size=20, font='times')
        pl.show_axes()
        pl.add_mesh(meshes[i], point_size=point_size, opacity=opacity, show_scalar_bar=False)
    pl.show()


if __name__ == '__main__':
    result_paths = glob.glob(r'./examples/shapenet/results/predict_err_ply/*/*')
    print(len(result_paths))

    case_id = random.randint(0, len(result_paths) // 3)

    gts = [p for p in result_paths if 'gt' in p]
    print(len(gts), case_id)

    gt_path = os.path.join(gts[case_id])
    print('Test case:', gt_path)

    pred_path = gt_path.replace('gt', 'pred')
    diff_path = gt_path.replace('gt', 'diff')
    meshes = []

    titles = ["Gt", "Pred", "Diff"]
    paths = [gt_path, pred_path, diff_path]

    for path in paths:
        meshes.append(pv.read(path))

    display_multi_meshes(meshes, titles)
