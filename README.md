

# Flatten-Surface
Python based tool for flattening mesh surfaces with compound curvatures using As-Rigid-As-Possible parameterization.


Quick notes to run the Open3D GUI for the flattening preprocessor.

Dependencies
- Install packages from `requirements.txt` (example):

```bash
pip install -r requirements.txt
```

Run
- From the `Flatten-Surface` folder run:

```bash
python gui_preprocessor.py
```

Features
- Load mesh (PLY/STL/OBJ...)
- Optional remeshing via PyMeshLab
- Flatten using ARAP (libigl)
- Visualize original and flattened mesh, color by strain
- Save flattened mesh (PLY/STL)

Notes
- Ensure `pyigl`/`pyigl` bindings and `pymeshlab` are installed and working on your platform.
- If you encounter crashes, try disabling remeshing or increasing target edge length.