# conda install -c conda-forge libstdcxx-ng=12

import pyvista as pv
import vtk

print(pv.Report(gpu=False))
w = vtk.vtkRenderWindow()
w.Render()
