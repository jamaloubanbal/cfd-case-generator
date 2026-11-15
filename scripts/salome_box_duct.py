"""
SALOME script: a box duct 1 x 0.2 x 0.2 m with face groups entree/sortie/parois, meshed with
NETGEN (tetrahedra) and exported as UNV. Runs without the GUI:

    ~/SALOME-9.16.0/salome -t scripts/salome_box_duct.py args:examples/meshes/box_duct_salome.unv
    ~/SALOME-9.16.0/salome killall

The committed examples/meshes/box_duct_salome.unv was produced this way with SALOME 9.16.0.
"""
import sys

import salome
salome.salome_init()

import GEOM  # noqa: F401
from salome.geom import geomBuilder
import SMESH
from salome.smesh import smeshBuilder

OUT = sys.argv[1] if len(sys.argv) > 1 else "box_duct_salome.unv"

geompy = geomBuilder.New()
box = geompy.MakeBoxDXDYDZ(1.0, 0.2, 0.2)
geompy.addToStudy(box, "box")

entree, sortie, parois = [], [], []
for face in geompy.SubShapeAll(box, geompy.ShapeType["FACE"]):
    x, _, _ = geompy.PointCoordinates(geompy.MakeCDG(face))
    (entree if abs(x) < 1e-9 else sortie if abs(x - 1.0) < 1e-9 else parois).append(face)

groups = {}
for name, faces in (("entree", entree), ("sortie", sortie), ("parois", parois)):
    g = geompy.CreateGroup(box, geompy.ShapeType["FACE"])
    geompy.UnionList(g, faces)
    geompy.addToStudyInFather(box, g, name)
    groups[name] = g

smesh = smeshBuilder.New()
mesh = smesh.Mesh(box, "box_duct")
algo = mesh.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)
params = algo.Parameters()
params.SetMaxSize(0.04)
params.SetMinSize(0.01)
params.SetSecondOrder(False)
ok = mesh.Compute()
print("compute ok:", ok, "nodes:", mesh.NbNodes(), "volumes:", mesh.NbVolumes(), "faces:", mesh.NbFaces())

for name, g in groups.items():
    mesh.GroupOnGeom(g, name, SMESH.FACE)
mesh.ExportUNV(OUT)
print("exported", OUT)
