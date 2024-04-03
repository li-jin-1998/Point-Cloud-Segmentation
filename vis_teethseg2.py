import glob
import os
import random

import pyvista as pv


def display_multi_meshes(meshes: list, titles=None, point_size=3, opacity=1.0):
    num = len(meshes)
    for i in range(num):
        pl.subplot(0, i)
        if i == num - 1:
            pl.add_checkbox_button_widget(toggle_vis, position=(500.0, 20.0), value=True)

        if titles is not None:
            pl.add_title(titles[i], font_size=20, font='times')
        pl.show_axes()
        pl.add_mesh(meshes[i], point_size=point_size, scalars=meshes[i].active_scalars, style="points", rgb=True,
                    opacity=opacity, show_scalar_bar=False)
        pl.view_xy()

    pl.show()


def toggle_vis(flag=0):
    pl.clear()
    case_id = random.randint(0, len(preds) - 1)
    pred_path = os.path.join(preds[case_id])
    print('Case {}:{}'.format(case_id, pred_path))
    src_path = os.path.join(src, os.path.basename(pred_path))
    gt_path = src_path.replace('.ply', '_label.ply')
    meshes = []

    titles = ["Src", "Gt", "Pred"]
    paths = [src_path, gt_path, pred_path]
    # print(paths)

    for path in paths:
        meshes.append(pv.read(path))
    display_multi_meshes(meshes, titles)


if __name__ == '__main__':
    src = '/mnt/algo_storage_server/PointCloudSeg/Dataset/data/'
    result_paths = glob.glob(r'./visualization/*')
    print(len(result_paths))
    preds = [p for p in result_paths if 'ply' in p and 'label' not in p]
    prev_id = []

    pl = pv.Plotter(shape=(1, 3))
    pl.set_background([0.9, 0.9, 0.9])
    pl.add_key_event("d", toggle_vis)
    toggle_vis(0)
    pl.show()
