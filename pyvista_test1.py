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


if __name__ == "__main__":
    file_path = './test2.ply'
    visualize_ply(file_path)
