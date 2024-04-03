import os
import glob
import random

import pyvista as pv


def display_multi_meshes(meshes: list, titles=None, point_size=3, opacity=0.9):
    num = len(meshes)
    for i in range(num):
        pl.subplot(0, i)
        if i == num - 1:
            pl.add_checkbox_button_widget(toggle_vis, position=(250.0, 20.0), value=True)

        if titles is not None:
            pl.add_title(titles[i], font_size=20, font='times')
        pl.show_axes()
        # pl.add_mesh(meshes[i], point_size=point_size, opacity=opacity, show_scalar_bar=False)
        pl.add_mesh(meshes[i], point_size=point_size, scalars=meshes[i].active_scalars, style="points", rgb=True,
                    opacity=opacity)
        pl.view_isometric()

    pl.show()


def toggle_vis(flag=0):
    pl.clear()
    case_id = random.randint(0, len(gts))

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


if __name__ == '__main__':
    result_paths = glob.glob(r'./examples/shapenet/results/predict_err_ply/*/*')
    print(len(result_paths))
    gts = [p for p in result_paths if 'gt' in p]

    pl = pv.Plotter(shape=(1, 3))
    pl.set_background([0.9, 0.9, 0.9])
    toggle_vis(0)
    pl.show()
