import os
import glob
import random

import pyvista as pv

result_paths = glob.glob(r'./examples/shapenet/results/predict_err_ply/*/*')
print(len(result_paths))

case_id = random.randint(0, len(result_paths) // 3)
point_size = 3
opacity = 0.8

gts = [p for p in result_paths if 'gt' in p]
print(len(gts))

gt_path = os.path.join(gts[case_id])
print('Test case:', gt_path)

pred_path = gt_path.replace('gt', 'pred')
diff_path = gt_path.replace('gt', 'diff')

gt_mesh = pv.read(gt_path)
pred_mesh = pv.read(pred_path)
diff_mesh = pv.read(diff_path)

pl = pv.Plotter(shape=(1, 3))
pl.set_background([0.9, 0.9, 0.9])


pl.subplot(0, 0)
def toggle_vis(flag):
    if flag:
        pl.add_mesh(gt_mesh, point_size=point_size, opacity=opacity, show_scalar_bar=False)

pl.add_checkbox_button_widget(toggle_vis,position=(300.0, 10.0), value=True)
pl.add_title('Gt', font_size=20, font='times')
pl.show_axes()
# pl.add_mesh(gt_mesh, point_size=point_size, opacity=opacity, show_scalar_bar=False)

pl.subplot(0, 1)
pl.add_title('Pred', font_size=20, font='times')
pl.show_axes()
pl.add_mesh(pred_mesh, point_size=point_size, opacity=opacity)

pl.subplot(0, 2)
pl.add_title('Diff', font_size=20, font='times')
pl.show_axes()
pl.add_mesh(diff_mesh, point_size=point_size, opacity=opacity)

pl.show()
