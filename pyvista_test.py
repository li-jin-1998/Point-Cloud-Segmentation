import random

import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def mesh_cmp_custom(mesh, name):
    """
    自定义色彩映射
    :param mesh: 输入mesh
    :param name: 比较数据的名字
    :return:
    """
    pts = mesh.points
    mesh[name] = pts[:, 1]
    # Define the colors we want to use
    blue = np.array([12 / 256, 238 / 256, 246 / 256, 1])
    black = np.array([11 / 256, 11 / 256, 11 / 256, 1])
    grey = np.array([189 / 256, 189 / 256, 189 / 256, 1])
    yellow = np.array([255 / 256, 247 / 256, 0 / 256, 1])
    red = np.array([1, 0, 0, 1])

    c_min = mesh[name].min()
    c_max = mesh[name].max()
    c_scale = c_max - c_min

    mapping = np.linspace(c_min, c_max, 256)
    newcolors = np.empty((256, 4))
    newcolors[mapping >= (c_scale * 0.8 + c_min)] = red
    newcolors[mapping < (c_scale * 0.8 + c_min)] = grey
    newcolors[mapping < (c_scale * 0.55 + c_min)] = yellow
    newcolors[mapping < (c_scale * 0.3 + c_min)] = blue
    newcolors[mapping < (c_scale * 0.1 + c_min)] = black

    # Make the colormap from the listed colors
    my_colormap = ListedColormap(newcolors)
    mesh.plot(scalars=name, cmap=my_colormap)


if __name__ == '__main__':
    mesh = pv.read('./test2.ply')
    print(mesh.points)
    colors = mesh.active_scalars
    print(colors[:, 0])
    print(mesh.n_cells)
    mesh.cell_data['color'] =colors
    # pv.plot(mesh.points, scalars=colors, window_size=(800,600),point_size=5,style="points", cpos='xy', rgb=True)

    p = pv.Plotter()

    sphere = pv.Sphere()
    p.add_mesh(sphere, name='sphere', show_edges=True)
    def toggle_vis(flag):
        p.clear()
        pos = random.randint(1, 100)
        for i in range(pos):
            p.add_checkbox_button_widget(toggle_vis, position=(random.randint(0, 1000), random.randint(0, 1000)), value=True)
        # sphere.SetVisibility(flag)

    p.add_checkbox_button_widget(toggle_vis, position=(300.0, 300.0), value=True)
    p.show()

    p = pv.Plotter()
    # def create_mesh(value):
    #     res = int(value)
    #     sphere = pv.Sphere(phi_resolution=res, theta_resolution=res)
    #     p.add_mesh(sphere, name='sphere', show_edges=True)
    #     return
    #
    #
    # p.add_slider_widget(create_mesh, [5, 100], title='Resolution')
    # p.show()
    # print(mesh.cell_data['colors'])
    # mesh.compute_normals()
    # normals = mesh.active_normals
    # mesh.plot(cpos='xy', scalars=colors[:,], cmap='jet',color='black')
    # mesh_cmp_custom(mesh, 'y_height')
    # p = pv.Plotter()
    # p.add_mesh(mesh, scalars=colors, style="points", rgb=True)
    # p.show_grid()
    # p.show()
