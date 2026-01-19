

# Flatten-Surface
Python based tool for flattening mesh surfaces.
This is intended for obtaining stencil shapes for cutting cloth to cover complex curvatures. 
The flattening algorithm consists of:
- Harmonic parameterization as initialization (mapping boundary and vertices to a circle)
- 20 iterations of Scalable Locally Injective Mapping, minimizing symmetric dirichlet isometric energy.
This provides results, in my experience, superior to certian CAD implementations.

Dependencies
- Install packages from `requirements.txt` (example):

```bash
pip install -r requirements.txt
```

Run
- From the `Flatten-Surface` folder run:

```bash
python gui.py
```

Features
- Load mesh (PLY/STL/OBJ...), or STEP surface
- Optional isotropic remeshing via PyMeshLab, recommended
- Flatten using SLIM algorithm (libigl)
- Visualize original and flattened mesh, colored by strain
- Save flattened mesh (STL/DXF/SVG)