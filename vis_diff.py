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
        pl.add_mesh(meshes[i], point_size=point_size, style="points", rgb=i != num - 1,
                    opacity=opacity, show_scalar_bar=False)
        pl.view_xy()

    pl.show()


def toggle_vis(flag=0):
    pl.clear()
    case_id = random.randint(0, len(preds) - 1)
    pred_path = os.path.join(preds[case_id])
    print('Case {}:{}'.format(case_id, pred_path))
    src_path = os.path.join('/mnt/algo_storage_server/PointCloudSeg/Dataset/data/', os.path.basename(pred_path))
    gt_path = src_path.replace('.ply', '_label.ply')
    diff_path = os.path.join('./diff', os.path.basename(gt_path))
    meshes = []

    titles = ["Src", "Gt", "Pred", "Diff"]
    paths = [src_path, gt_path, pred_path, diff_path]
    # print(paths)

    for path in paths:
        if not os.path.exists(path):
            print('The file {} not exist.'.format(path))
            return
        meshes.append(pv.read(path))
    display_multi_meshes(meshes, titles)


def save_screenshot():
    pl.screenshot('./screenshot.png')
    print('save screenshot')

if __name__ == '__main__':
    result_paths = glob.glob(r'./visualization/*')
    print(len(result_paths))
    preds = [p for p in result_paths if 'ply' in p and 'label' not in p]

    pl = pv.Plotter(shape=(1, 4))
    pl.set_background([0.9, 0.9, 0.9])
    pl.add_key_event("d", toggle_vis)
    pl.add_key_event("s", save_screenshot)
    toggle_vis(0)
    pl.show()
