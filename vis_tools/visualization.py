import pyvista as pv


class VisualizationPoints:
    """
    Visualize a list of points with different colors.
    """

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
