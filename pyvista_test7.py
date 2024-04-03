import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
import pyvista as pv


def select_path():
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(None, "Select File", ".", "")
    return file_path


file_path = select_path()
if file_path:
    print(file_path)
    try:
        mesh = pv.read(file_path)
        p = pv.Plotter()
        p.add_mesh(mesh, scalars=mesh.active_scalars, style="points", rgb=True)
        p.show(cpos='xy')
    except ValueError:
        print('error')
else:
    print("No file selected.")
