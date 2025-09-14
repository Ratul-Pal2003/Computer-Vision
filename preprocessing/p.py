import vtk
import nibabel as nib
import pyvista as pv
import numpy as np

# ---- Load MRI (example file) ----
img_file = r"../raw data/Oasis-2/part1_extracted/OAS2_RAW_PART1/OAS2_0001_MR1/RAW/mpr-1.nifti.img"
img = nib.load(img_file)
data = img.get_fdata()

pv.set_jupyter_backend(None)  # disables notebook mode, forces VTK window

from vtkmodules.vtkCommonDataModel import vtkImageData

# Create vtkImageData manually
image = vtkImageData()
nx, ny, nz = data.shape
image.SetDimensions(nx, ny, nz)
image.SetSpacing(img.header.get_zooms()[:3])
image.SetOrigin(0, 0, 0)

# Convert numpy → vtk
import numpy as np
from vtkmodules.util import numpy_support
vtk_data = numpy_support.numpy_to_vtk(
    num_array=data.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
)
image.GetPointData().SetScalars(vtk_data)

# Wrap into pyvista object
grid = pv.wrap(image)

# Plot
plotter = pv.Plotter()
plotter.add_volume(grid, cmap="gray", opacity="sigmoid")
plotter.show()
