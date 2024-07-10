import glob
import os
import random

import pyvista as pv


class VisualizationMeshes:
    """
    Visualize multiple meshes
    """
    def __init__(self, pred_paths, titles, num_meshes=3, point_size=3, opacity=1.0):
        self.pl = None
        self.meshes = []
        self.titles = titles
        self.pred_paths = pred_paths
        self.pred_path = None
        self.point_size = point_size
        self.opacity = opacity
        self.num_meshes = num_meshes
        self.init_plotter()

    def init_plotter(self):
        self.pl = pv.Plotter(shape=(1, self.num_meshes))
        self.pl.set_background([0.9, 0.9, 0.9])
        self.pl.add_key_event("d", self.show)
        self.pl.add_key_event("s", self.save_screenshot)

    def display(self):
        for i in range(self.num_meshes):
            self.pl.subplot(0, i)
            if i == self.num_meshes - 1:
                self.pl.add_checkbox_button_widget(self.show, position=(500.0, 20.0), value=True)

            if self.titles is not None:
                self.pl.add_title(self.titles[i], font_size=20, font='times')
            self.pl.show_axes()
            self.pl.add_mesh(self.meshes[i], point_size=self.point_size, style="points", rgb=True,
                             opacity=self.opacity, show_scalar_bar=False)
            self.pl.view_xy()

        self.pl.show()

    def save_screenshot(self):
        # os.makedirs('./screenshot',exist_ok=True)
        save_path = os.path.join('../screenshot', os.path.basename(self.pred_path)).replace('ply', 'png')
        if os.path.exists(save_path):
            os.remove(save_path)
        self.pl.screenshot(save_path)
        print('Saved to {}'.format(save_path))

    def show(self, flag=0):
        self.pl.clear()
        case_id = random.randint(0, len(self.pred_paths) - 1)
        self.pred_path = os.path.join(self.pred_paths[case_id])
        print('Case {}:{}'.format(case_id, self.pred_path))
        src_path = os.path.join(src, os.path.basename(self.pred_path))
        gt_path = src_path.replace('.ply', '_label.ply')

        paths = [src_path, gt_path, self.pred_path]
        for path in paths:
            if not os.path.exists(path):
                print('The file {} not exist.'.format(path))
                return
        self.meshes = [pv.read(path) for path in paths]
        self.display()


if __name__ == '__main__':
    src = '/mnt/algo_storage_server/PointCloudSeg/Dataset/data/'
    result_paths = glob.glob(r'./visualization/*')
    print(len(result_paths))
    preds = [p for p in result_paths if 'ply' in p and 'label' not in p]

    vis = VisualizationMeshes(pred_paths=preds, titles=["Src", "Gt", "Pred"])
    vis.show()
