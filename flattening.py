import open3d as o3d
import numpy as np
import igl
import pymeshlab
import matplotlib

def unique_edges(F):
    edges = np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges

def strain_to_rgb(strain, smax=0.02):
    if smax is None:
        smax = np.max(np.abs(strain))
        if smax == 0:
            smax = 1e-6
    strain_clipped = np.clip(strain, -smax, smax)
    t = (strain_clipped + smax) / (2*smax)
    cmap = matplotlib.colormaps["RdBu"]
    colors = cmap(t)[:, :3]
    return colors

def load_mesh(path, remeshing=True, target_edge_length=1, iterations=20):
    """Load mesh and remesh isotropically using PyMeshLab."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    
    # Clean mesh
    ms.apply_filter('meshing_remove_duplicate_faces')
    ms.apply_filter('meshing_remove_duplicate_vertices')
    ms.apply_filter('meshing_remove_null_faces')
    ms.apply_filter('meshing_remove_unreferenced_vertices')
    
    if remeshing:
        # Isotropic explicit remeshing
        ms.apply_filter('meshing_isotropic_explicit_remeshing',
                        targetlen=pymeshlab.PureValue(target_edge_length),
                        iterations=iterations) 
    
    mesh = ms.current_mesh()
    v = mesh.vertex_matrix()
    f = mesh.face_matrix()
    
    return mesh, np.array(v), np.array(f)

def flatten_mesh_arap(v, f):

    # Harmonic parameterization as initialization
    bnd = igl.boundary_loop(f)
    bnd_uv = igl.map_vertices_to_circle(v, bnd)
    uv = igl.harmonic(v,f,bnd,bnd_uv,1)

    #SLIM flattening 
    b = np.zeros(0, dtype=np.int32)
    bc = np.zeros((0,2), dtype=np.float64)
    slim_data = igl.slim_precompute(v, f, uv, igl.MappingEnergyType.SYMMETRIC_DIRICHLET, b, bc)
    uv_flat = igl.slim_solve(slim_data, iter_num=30)

    # Align triangle 0, vertex 0 to original XY
    tri_id = 0
    corner_id = f[tri_id, 0]
    translation = v[corner_id, :2] - uv_flat[corner_id]
    uv_flat_aligned = uv_flat + translation

    return uv, uv_flat_aligned

def realign_flattened_mesh(v_orig, uv_flat):
    """
    Realign a flattened mesh (uv_flat) to the original mesh (v_orig) in XY plane.
    The flattened mesh is translated and rotated so that it best aligns in XY,
    and lies on the global XY plane (Z=0).

    Args:
        v_orig: (N,3) original 3D mesh vertices
        uv_flat: (N,2) flattened vertices from ARAP

    Returns:
        uv_aligned: (N,2) flattened vertices aligned to original XY
        R: 2x2 rotation matrix applied
        t: 2-element translation vector applied
    """
    # Step 1: Center both meshes in XY
    orig_xy = v_orig[:, :2]
    flat_xy = uv_flat
    orig_center = orig_xy.mean(axis=0)
    flat_center = flat_xy.mean(axis=0)
    flat_xy_centered = flat_xy - flat_center

    # Step 2: Compute optimal 2D rotation using Procrustes method
    # R = argmin || R*flat_xy_centered - (orig_xy - orig_center) ||_F
    H = flat_xy_centered.T @ (orig_xy - orig_center)
    U, S, VT = np.linalg.svd(H)
    R = VT.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        VT[1,:] *= -1
        R = VT.T @ U.T

    # Step 3: Rotate flattened mesh
    uv_rot = (R @ flat_xy_centered.T).T

    # Step 4: Translate to match original XY centroid
    t = orig_center
    uv_aligned = uv_rot + t
    #print("Applied rotation:\n", R)
    #print("Applied translation:\n", t)

    return uv_aligned, R, t

def calculate_vertex_strain(v, f, uv_flat):
    edges = unique_edges(f)
    L0 = np.linalg.norm(v[edges[:,0]] - v[edges[:,1]], axis=1)
    L1 = np.linalg.norm(uv_flat[edges[:,0]] - uv_flat[edges[:,1]], axis=1)
    edge_strain = (L1 - L0) / L0

    vertex_strain = np.zeros(len(v))
    counts = np.zeros(len(v))
    for i, (a, b) in enumerate(edges):
        s = edge_strain[i]
        vertex_strain[a] += s
        vertex_strain[b] += s
        counts[a] += 1
        counts[b] += 1
    vertex_strain /= counts
    return vertex_strain 

def build_o3d_mesh_from_vf(v, f, vertex_colors=None):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(v))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(f))
    if vertex_colors is not None:
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors))
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d 

if __name__ == "__main__":
    mesh_path = r"/home/herman/Downloads/001196.STL"

    target_edge_length = 2.0
    mesh, v, f = load_mesh(mesh_path, target_edge_length, iterations=20)

    uv = None
    while uv is None and target_edge_length < 10.0:
        try:
            uv, uv_flat = flatten_mesh_arap(v, f)
            # should add more quality checks here
        except:
            target_edge_length += 0.2
            mesh, v, f = load_mesh(mesh_path, target_edge_length=target_edge_length, iterations=20)
            print(f"Retrying flattening with target edge length: {target_edge_length}")

    # Realign flattened mesh
    uv_flat, R2d, t2d = realign_flattened_mesh(v, uv_flat)

    # 3D flattened mesh
    V_flat_3d = np.column_stack([uv_flat, np.zeros(len(uv_flat))])
    mesh_flat_o3d = build_o3d_mesh_from_vf(V_flat_3d, f)

    # Original mesh
    mesh_orig_o3d = build_o3d_mesh_from_vf(v, f)

    # Optional: Color by strain
    vertex_strain = calculate_vertex_strain(v, f, uv_flat)
    mesh_orig_o3d.vertex_colors = o3d.utility.Vector3dVector(strain_to_rgb(vertex_strain))
    mesh_flat_o3d.vertex_colors = o3d.utility.Vector3dVector(strain_to_rgb(vertex_strain))

    # Visualize
    o3d.visualization.draw_geometries([
        mesh_orig_o3d,
        mesh_flat_o3d
    ], window_name="Original + Flattened Mesh", mesh_show_wireframe=False)


