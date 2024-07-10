import pyvista as pv


def visualize_ply(file_path):
    try:
        mesh = pv.read(file_path)
    except Exception as e:
        print(f"无法读取文件: {file_path}。\n错误: {str(e)}")
        return

    p = pv.Plotter()
    p.add_mesh(mesh, style="points", rgb=True)
    p.show(cpos='xy')
    # 捕获渲染结果并保存为图像文件
    p.screenshot('screenshot.png')


if __name__ == "__main__":
    # print(pv.Report(gpu=False))
    file_path = '../test2.ply'
    visualize_ply(file_path)
    mesh = pv.read(file_path)
    pv.plot(mesh.points, scalars=mesh.active_scalars, point_size=5, cpos='xy', style="points", rgb=True)
