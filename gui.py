import os
import threading
import traceback
import logging
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
import tkinter as tk
from tkinter import filedialog

from flattening import (
    load_mesh,
    remesh_mesh,
    flatten_mesh,
    realign_flattened_mesh,
    calculate_vertex_strain,
    strain_to_rgb,
    build_o3d_mesh_from_vf
)

# setup logging
LOG_PATH = os.path.join(os.path.dirname(__file__), "gui_log.txt")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
)

def _log_exc(context=""):
    logging.error("%s\n%s", context, traceback.format_exc())

class FlatteningGUI:
    def __init__(self, width=1200, height=700):
        self.window = gui.Application.instance.create_window(
            "Flatten Surface - GUI", width, height
        )

        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")
        try:
            if os.path.exists(icon_path):
                self.window.set_icon(o3d.geometry.Image(icon_path))
        except Exception:
            pass  # Icon loading is optional

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        # Use a light gray background for better contrast

        self.scene_widget.scene.set_background([0.3, 0.3, 0.3, 1.0])


        self.v = None
        self.f = None
        self.mesh_ms = None
        self.mesh_orig = None
        self.mesh_flat = None
        self.wireframe_orig = None
        self.wireframe_flat = None
        self.mesh_path = None

        # Controls
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self.load_button = gui.Button("Load Mesh")
        self.load_button.horizontal_padding_em = 0.5
        self.load_button.vertical_padding_em = 0
        self.load_button.set_on_clicked(self._on_load)

        self.target_len = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_len.set_value(1.0)
        self.target_len.set_limits(0.001, 1000.0)

        self.iter_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.iter_edit.set_value(20)
        self.iter_edit.set_limits(1, 1000)

        self.remesh_button = gui.Button("Remesh")
        self.remesh_button.set_on_clicked(self._on_remesh)
        self.remesh_button.enabled = False

        self.flatten_button = gui.Button("Flatten")
        self.flatten_button.set_on_clicked(self._on_flatten)

        self.show_orig = gui.Checkbox("Show Original")
        self.show_orig.checked = True
        self.show_orig.set_on_checked(self._on_toggle_show)

        self.show_flat = gui.Checkbox("Show Flattened")
        self.show_flat.checked = True
        self.show_flat.set_on_checked(self._on_toggle_show)

        self.wireframe = gui.Checkbox("Wireframe")
        self.wireframe.checked = False
        self.wireframe.set_on_checked(self._on_toggle_wireframe)

        self.save_button = gui.Button("Save Flattened")
        self.save_button.set_on_clicked(self._on_save)
        self.save_button.enabled = False

        self.strain_legend = gui.Label("Strain Range: 0.0000% to 0.0000%")
        self.strain_legend.text_color = gui.Color(0.5, 0.5, 0.5)

        self.status = gui.Label("Ready.")

        self.panel.add_child(self.load_button)
        self.panel.add_child(gui.Label("Target edge length:"))
        self.panel.add_child(self.target_len)
        self.panel.add_child(gui.Label("Remesh iterations:"))
        self.panel.add_child(self.iter_edit)
        self.panel.add_child(self.remesh_button)
        self.panel.add_child(self.flatten_button)
        self.panel.add_child(self.show_orig)
        self.panel.add_child(self.show_flat)
        self.panel.add_child(self.wireframe)
        self.panel.add_child(self.save_button)
        self.panel.add_child(self.strain_legend)
        self.panel.add_child(self.status)

        # Layout
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = int(r.width * 0.28)
        # Place the control panel on the left and the scene on the right
        self.panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x + panel_width, r.y, r.width - panel_width, r.height)

    def _on_load(self):
        # Open file dialog in a separate thread to avoid blocking the main thread
        # and to work around Open3D FileDialog issues on Windows
        self.load_button.enabled = False
        
        def open_file_dialog():
            try:                
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                filename = filedialog.askopenfilename(
                    title="Open mesh",
                    filetypes=[
                        ("Mesh files", "*.stl *.ply *.obj *.off *.gltf *.glb"),
                        ("Surface files", "*.step *.stp"),
                        ("All files", "*.*"),
                    ],
                )
                root.destroy()
                
                if filename:
                    # Update GUI on main thread
                    def on_main_thread():
                        self.status.text = f"Loading {os.path.basename(filename)}..."
                        self._load_mesh_from_path(filename)
                    gui.Application.instance.post_to_main_thread(self.window, on_main_thread)
                else:
                    # User cancelled
                    def on_main_thread():
                        self.load_button.enabled = True
                    gui.Application.instance.post_to_main_thread(self.window, on_main_thread)
                    
            except Exception as e:
                _log_exc("Error in file dialog thread")
                def on_main_thread():
                    self.load_button.enabled = True
                
                gui.Application.instance.post_to_main_thread(self.window, on_main_thread)
        
        # Run file dialog in background thread
        threading.Thread(target=open_file_dialog, daemon=True).start()


    def _load_mesh_from_path(self, path):
        
        def worker_load():
            try:
                ms, v, f = load_mesh(path)
                def finish():
                    try:
                        # remove previous geometry if present
                        try:
                            self.scene_widget.scene.remove_geometry("orig")
                        except Exception:
                            pass

                        self.v = v
                        self.f = f
                        self.mesh_ms = ms
                        self.mesh_path = path
                        mesh_o3d = build_o3d_mesh_from_vf(v, f)
                        self.mesh_orig = mesh_o3d
                        self.wireframe_orig = self._create_wireframe(mesh_o3d)
                        self._add_geometry("orig", mesh_o3d)
                        if self.wireframe_orig is not None:
                            self._add_wireframe("wireframe_orig", self.wireframe_orig)
                        self.remesh_button.enabled = True
                        self.status.text = f"Loaded: {os.path.basename(path)}"
                    except Exception:
                        _log_exc("Error finishing load on main thread")
                    finally:
                        self.load_button.enabled = True

                gui.Application.instance.post_to_main_thread(self.window, finish)
            except Exception:
                _log_exc("Exception during background load")

                def fail():
                    self.status.text = f"Load failed (see gui_log.txt)"
                    self.load_button.enabled = True

                gui.Application.instance.post_to_main_thread(self.window, fail)
                
        threading.Thread(target=worker_load, daemon=True).start()

    def _on_remesh(self):
        if self.v is None or self.f is None:
            self.status.text = "No mesh loaded."
            return

        self.status.text = "Remeshing..."
        self.remesh_button.enabled = False

        def worker_remesh():
            try:
                target = self.target_len.double_value
                iters = self.iter_edit.int_value
                #logging.debug(f"Starting remesh: target={target}, iters={iters}")
                if not self.mesh_path:
                    raise RuntimeError("No mesh path available for remeshing")
                _, v_remesh, f_remesh = remesh_mesh(self.mesh_path, target_edge_length=target, iterations=iters)
                #logging.debug("Remesh completed successfully")
                
                def finish():
                    try:
                        # Update mesh data
                        self.v = v_remesh
                        self.f = f_remesh
                        # Remove old mesh and wireframe
                        try:
                            self.scene_widget.scene.remove_geometry("orig")
                        except Exception:
                            pass
                        try:
                            self.scene_widget.scene.remove_geometry("wireframe_orig")
                        except Exception:
                            pass
                        
                        mesh_o3d = build_o3d_mesh_from_vf(v_remesh, f_remesh)
                        self.mesh_orig = mesh_o3d
                        self.wireframe_orig = self._create_wireframe(mesh_o3d)
                        self._add_geometry("orig", mesh_o3d)
                        if self.wireframe_orig is not None:
                            self._add_wireframe("wireframe_orig", self.wireframe_orig)
                        
                        self.status.text = "Remeshing finished."
                    except Exception:
                        _log_exc("Error finishing remesh on main thread")
                    finally:
                        self.remesh_button.enabled = True

                gui.Application.instance.post_to_main_thread(self.window, finish)
            except Exception:
                _log_exc("Exception during remesh")

                def fail():
                    self.status.text = "Remesh failed (see gui_log.txt)"
                    self.remesh_button.enabled = True

                gui.Application.instance.post_to_main_thread(self.window, fail)

        threading.Thread(target=worker_remesh, daemon=True).start()


    def _on_flatten(self):
        if self.v is None or self.f is None:
            self.status.text = "No mesh loaded."
            return

        self.status.text = "Flattening..."
        self.flatten_button.enabled = False

        def worker():
            try:
                uv, uv_flat = flatten_mesh(self.v, self.f)
                # realign
                uv_flat_aligned, R, t = realign_flattened_mesh(self.v, uv_flat)

                V_flat_3d = np.column_stack([uv_flat_aligned, np.zeros(len(uv_flat_aligned))])
                mesh_flat_o3d = build_o3d_mesh_from_vf(V_flat_3d, self.f)

                vertex_strain = calculate_vertex_strain(self.v, self.f, uv_flat_aligned)
                # Get actual strain min/max for colorbar display
                strain_min = np.min(vertex_strain) * 100 # convert to percentage
                strain_max = np.max(vertex_strain) * 100 # convert to percentage
                strain_range = max(abs(strain_min), abs(strain_max))
                
                colors = strain_to_rgb(vertex_strain)
                mesh_flat_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)

                # Also color original by strain
                mesh_orig_colored = build_o3d_mesh_from_vf(self.v, self.f, vertex_colors=colors)

                def finish():
                    # remove old if present
                    try:
                        self.scene_widget.scene.remove_geometry("flat")
                    except Exception:
                        pass
                    try:
                        self.scene_widget.scene.remove_geometry("orig")
                    except Exception:
                        pass

                    self.mesh_flat = mesh_flat_o3d
                    self.mesh_orig = mesh_orig_colored
                    self.wireframe_orig = self._create_wireframe(mesh_orig_colored)
                    self.wireframe_flat = self._create_wireframe(mesh_flat_o3d)
                    self._add_geometry("orig", mesh_orig_colored)
                    self._add_geometry("flat", mesh_flat_o3d)
                    if self.wireframe_orig is not None:
                        self._add_wireframe("wireframe_orig", self.wireframe_orig)
                    if self.wireframe_flat is not None:
                        self._add_wireframe("wireframe_flat", self.wireframe_flat)
                    # Update strain legend with actual values
                    self.strain_legend.text = f"Strain Range: {strain_min:.4f}% (Red) to {strain_max:.4f}% (Blue)"

                    
                    self.save_button.enabled = True
                    self.flatten_button.enabled = True
                    self.status.text = "Flattening finished."

                gui.Application.instance.post_to_main_thread(self.window, finish)
            except Exception as e:
                _log_exc("Exception during flatten worker")

                def fail():
                    self.status.text = f"Flatten failed (see gui_log.txt)"
                    self.flatten_button.enabled = True

                gui.Application.instance.post_to_main_thread(self.window, fail)

        threading.Thread(target=worker, daemon=True).start()

    def _add_geometry(self, name, mesh):
        try:
            # Remove any existing geometry with the same name to avoid duplicates
            try:
                self.scene_widget.scene.remove_geometry(name)
            except Exception:
                pass

            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            # Convert to TriangleMesh for renderer
            self.scene_widget.scene.add_geometry(name, mesh, material)

            bbox = mesh.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(60, bbox, bbox.get_center())
        except Exception:
            _log_exc(f"Error adding geometry '{name}'")

    def _add_wireframe(self, name, line_set):
        try:
            # Remove any existing wireframe with the same name first
            try:
                self.scene_widget.scene.remove_geometry(name)
            except Exception:
                pass

            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.line_width = 1.5
            self.scene_widget.scene.add_geometry(name, line_set, material)
            # Initially hide wireframe
            self.scene_widget.scene.show_geometry(name, False)
        except Exception:
            _log_exc(f"Error adding wireframe '{name}'")

    def _on_toggle_show(self, checked):
        try:
            if self.mesh_orig is not None:
                self.scene_widget.scene.show_geometry("orig", self.show_orig.checked)
            if self.mesh_flat is not None:
                self.scene_widget.scene.show_geometry("flat", self.show_flat.checked)
        except Exception:
            pass

    def _on_toggle_wireframe(self, checked):
        try:
            # Toggle wireframe visibility
            if self.wireframe.checked:
                # Show wireframes
                if self.wireframe_orig is not None:
                    self.scene_widget.scene.show_geometry("wireframe_orig", True)
                if self.wireframe_flat is not None:
                    self.scene_widget.scene.show_geometry("wireframe_flat", True)
            else:
                # Hide wireframes
                try:
                    self.scene_widget.scene.show_geometry("wireframe_orig", False)
                except Exception:
                    pass
                try:
                    self.scene_widget.scene.show_geometry("wireframe_flat", False)
                except Exception:
                    pass
        except Exception:
            pass
    
    def _create_wireframe(self, mesh):
        """Create a wireframe representation from a mesh."""
        try:
            edges = []
            for triangle in np.asarray(mesh.triangles):
                edges.extend([
                    [triangle[0], triangle[1]],
                    [triangle[1], triangle[2]],
                    [triangle[2], triangle[0]]
                ])
            
            lines = o3d.geometry.LineSet()
            lines.points = mesh.vertices
            lines.lines = o3d.utility.Vector2iVector(np.array(edges))
            lines.paint_uniform_color([0.0, 0.0, 0.0])  # Black wireframe
            return lines
        except Exception:
            return None

    def _on_save(self):
        if self.mesh_flat is None:
            self.status.text = "No flattened mesh to save."
            return

        def save_file_dialog():
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                filename = filedialog.asksaveasfilename(
                    title="Save flattened mesh",
                    defaultextension=".stl",
                    filetypes=[
                        ("Vector files", "*.dxf *.svg"),
                        ("Mesh files", "*.stl *.ply *.obj *.off *.gltf *.glb"),
                        ("Surface files", "*.step *.stp"),
                        ("All files", "*.*"),
                    ],
                )
                root.destroy()
                
                if filename:
                    # Save on main thread
                    def on_main_thread():
                        try:
                            o3d.io.write_triangle_mesh(filename, self.mesh_flat)
                            self.status.text = f"Saved {os.path.basename(filename)}"
                        except Exception as e:
                            _log_exc("Error saving mesh")
                            self.status.text = f"Save failed: {e}"
                    
                    gui.Application.instance.post_to_main_thread(self.window, on_main_thread)
                    
            except Exception as e:
                _log_exc("Error in save file dialog thread")
                gui.Application.instance.post_to_main_thread(self.window, on_main_thread)
        
        # Run file dialog in background thread
        threading.Thread(target=save_file_dialog, daemon=True).start()


def main():
    try:
        gui.Application.instance.initialize()
        app = FlatteningGUI()
        gui.Application.instance.run()
    except Exception:
        _log_exc("Unhandled exception in main loop")


if __name__ == "__main__":
    main()
