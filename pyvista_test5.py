import pyvista as pv

if __name__ == '__main__':
    mesh = pv.read('./test2.ply')
    print(mesh.points)
    colors = mesh.active_scalars
    print(colors)
    # pv.plot(mesh.points, scalars=colors, point_size=5, style="points", cpos='xy', rgb=True)
    p = pv.Plotter()
    p.add_mesh(mesh, scalars=colors, style="points", rgb=True)
    p.add_background_image("test.png")
    p.show(cpos='xy')
