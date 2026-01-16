import os
import threading
import traceback
import logging
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from flattening import (
    load_mesh,
    flatten_mesh_arap,
    realign_flattened_mesh,
    calculate_vertex_strain,
    strain_to_rgb,
    build_o3d_mesh_from_vf,
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

def _show_error_dialog(window, title, message):
    def _show():
        dlg = gui.Dialog(title)
        lbl = gui.Label(message)
        btn = gui.Button("OK")
        btn.set_on_clicked(lambda: dlg.close())
        dlg.add_child(lbl)
        dlg.add_child(btn)
        window.show_dialog(dlg)

    gui.Application.instance.post_to_main_thread(window, _show)


class FlatteningGUI:
    def __init__(self, width=1200, height=700):
        self.window = gui.Application.instance.create_window(
            "Flatten Surface - GUI", width, height
        )

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)

        self.v = None
        self.f = None
        self.mesh_ms = None
        self.mesh_orig = None
        self.mesh_flat = None

        # Controls
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self.load_button = gui.Button("Load Mesh")
        self.load_button.horizontal_padding_em = 0.5
        self.load_button.vertical_padding_em = 0
        self.load_button.set_on_clicked(self._on_load)

        self.remesh_checkbox = gui.Checkbox("Remesh (PyMeshLab)")
        self.remesh_checkbox.checked = True

        self.target_len = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_len.set_value(1.0)
        self.target_len.set_limits(0.001, 1000.0)

        self.iter_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.iter_edit.set_value(20)
        self.iter_edit.set_limits(1, 1000)

        self.flatten_button = gui.Button("Flatten (ARAP)")
        self.flatten_button.set_on_clicked(self._on_flatten)

        self.show_orig = gui.Checkbox("Show Original")
        self.show_orig.checked = True
        self.show_orig.set_on_checked(self._on_toggle_show)

        self.show_flat = gui.Checkbox("Show Flattened")
        self.show_flat.checked = True
        self.show_flat.set_on_checked(self._on_toggle_show)

        self.save_button = gui.Button("Save Flattened")
        self.save_button.set_on_clicked(self._on_save)
        self.save_button.enabled = False

        self.status = gui.Label("Ready.")

        self.panel.add_child(self.load_button)
        self.panel.add_child(self.remesh_checkbox)
        self.panel.add_child(gui.Label("Target edge length:"))
        self.panel.add_child(self.target_len)
        self.panel.add_child(gui.Label("Remesh iterations:"))
        self.panel.add_child(self.iter_edit)
        self.panel.add_child(self.flatten_button)
        self.panel.add_child(self.show_orig)
        self.panel.add_child(self.show_flat)
        self.panel.add_child(self.save_button)
        self.panel.add_child(self.status)

        # Layout
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = int(r.width * 0.28)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)

    def _on_load(self):
        # Prefer tkinter file dialog on Windows/Linux (more reliable on many setups).
        # Fall back to Open3D's FileDialog if tkinter isn't available or fails.
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            self.load_button.enabled = False
            filename = filedialog.askopenfilename(
                title="Open mesh",
                filetypes=[
                    #("Mesh files", "*.ply *.stl *.obj *.off *.gltf *.glb"),
                    ("All files", "*.*"),
                ],
            )
            root.destroy()
            self.load_button.enabled = True
            if filename:
                self._load_mesh_from_path(filename)
            return
        except Exception:
            _log_exc("tkinter file dialog failed, falling back to Open3D FileDialog")


    def _load_mesh_from_path(self, path):
        self.status.text = f"Loading {os.path.basename(path)} ..."
        self.load_button.enabled = False

        def worker_load():
            try:
                remesh = self.remesh_checkbox.checked
                target = self.target_len.double_value
                iters = self.iter_edit.int_value
                ms, v, f = load_mesh(path, remeshing=remesh, target_edge_length=target, iterations=iters)

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
                        mesh_o3d = build_o3d_mesh_from_vf(v, f)
                        self.mesh_orig = mesh_o3d
                        self._add_geometry("orig", mesh_o3d)
                        self.status.text = f"Loaded: {os.path.basename(path)}"
                    except Exception:
                        _log_exc("Error finishing load on main thread")
                        _show_error_dialog(self.window, "Load Error", "An error occurred while adding mesh to the scene. See gui_log.txt")
                    finally:
                        self.load_button.enabled = True

                gui.Application.instance.post_to_main_thread(self.window, finish)
            except Exception:
                _log_exc("Exception during background load")

                def fail():
                    self.status.text = f"Load failed (see gui_log.txt)"
                    self.load_button.enabled = True
                    _show_error_dialog(self.window, "Load Failed", "Loading the mesh failed. See gui_log.txt for details.")

                gui.Application.instance.post_to_main_thread(self.window, fail)

        threading.Thread(target=worker_load, daemon=True).start()

    def _on_flatten(self):
        if self.v is None or self.f is None:
            self.status.text = "No mesh loaded."
            return

        self.status.text = "Flattening..."
        self.flatten_button.enabled = False

        def worker():
            try:
                uv, uv_flat = flatten_mesh_arap(self.v, self.f)
                # realign
                uv_flat_aligned, R, t = realign_flattened_mesh(self.v, uv_flat)

                V_flat_3d = np.column_stack([uv_flat_aligned, np.zeros(len(uv_flat_aligned))])
                mesh_flat_o3d = build_o3d_mesh_from_vf(V_flat_3d, self.f)

                vertex_strain = calculate_vertex_strain(self.v, self.f, uv_flat_aligned)
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
                    self._add_geometry("orig", mesh_orig_colored)
                    self._add_geometry("flat", mesh_flat_o3d)
                    self.save_button.enabled = True
                    self.flatten_button.enabled = True
                    self.status.text = "Flattening finished."

                gui.Application.instance.post_to_main_thread(self.window, finish)
            except Exception as e:
                _log_exc("Exception during flatten worker")

                def fail():
                    self.status.text = f"Flatten failed (see gui_log.txt)"
                    self.flatten_button.enabled = True
                    _show_error_dialog(self.window, "Flatten Failed", "Flattening failed. See gui_log.txt for details.")

                gui.Application.instance.post_to_main_thread(self.window, fail)

        threading.Thread(target=worker, daemon=True).start()

    def _add_geometry(self, name, mesh):
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        # Convert to TriangleMesh for renderer
        self.scene_widget.scene.add_geometry(name, mesh, material)

        bbox = mesh.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bbox, bbox.get_center())

    def _on_toggle_show(self, checked):
        try:
            if self.mesh_orig is not None:
                self.scene_widget.scene.show_geometry("orig", self.show_orig.checked)
            if self.mesh_flat is not None:
                self.scene_widget.scene.show_geometry("flat", self.show_flat.checked)
        except Exception:
            pass

    def _on_save(self):
        if self.mesh_flat is None:
            self.status.text = "No flattened mesh to save."
            return

        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Save flattened mesh", self.window.theme)
        dlg.add_filter("PLY", ["*.ply"]) 
        dlg.add_filter("STL", ["*.stl"]) 

        def on_cancel():
            dlg.close()

        def on_done(filename):
            dlg.close()
            try:
                o3d.io.write_triangle_mesh(filename, self.mesh_flat)
                self.status.text = f"Saved {os.path.basename(filename)}"
            except Exception as e:
                self.status.text = f"Save failed: {e}"

        dlg.set_on_cancel(on_cancel)
        dlg.set_on_done(on_done)
        self.window.show_dialog(dlg)


def main():
    try:
        gui.Application.instance.initialize()
        app = FlatteningGUI()
        gui.Application.instance.run()
    except Exception:
        _log_exc("Unhandled exception in main loop")
        # If window exists, try to show dialog
        try:
            _show_error_dialog(gui.Application.instance.active_window, "Fatal Error", "Unhandled exception. See gui_log.txt")
        except Exception:
            pass


if __name__ == "__main__":
    main()
