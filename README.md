

# Flatten-Surface
Python based tool for flattening mesh surfaces with compound curvatures using Scalable Locally Injective Mapping for strain reduction. 


Quick notes to run the Open3D GUI for the flattener.

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
- Optional remeshing via PyMeshLab
- Flatten using SLIM algorithm (libigl)
- Visualize original and flattened mesh, color by strain
- Save flattened mesh (STL), DXF/STEP coming soon
