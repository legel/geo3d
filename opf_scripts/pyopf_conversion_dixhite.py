'''
Convert an Open Photogrammetry Format (OPF) file and directory to nerfstudio format
using the Pix4D OPF python library (pyopf).
'''

from pyopf.io import load

from pyopf.resolve import resolve
from pyopf.uid64 import Uid64

# Path to the example project file.
project_path = "dixhite_geofusion_opf/project.opf"

# Load the json data and resolve the project, i.e. load the project items as named attributes.
project = load(project_path)
project = resolve(project)

# print("Project")
# print(project)

# Many objects are optional in OPF. If they are missing, they are set to None.
if project.calibration is None:
    print("No calibration data.")
    exit(1)
