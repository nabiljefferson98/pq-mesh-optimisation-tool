"""
src/visualisation/interactive_optimisation.py

Interactive 3D viewer and real-time optimisation controller for PQ mesh
optimisation.

This module provides the primary end-user interface for the project. It opens
a Polyscope 3D viewer showing the original design mesh and the current working
mesh side by side, with an ImGui control panel that allows the user to adjust
energy weights, run the optimiser, inspect individual panels and vertices,
and export results -- all without writing a single line of code.

Architecture
------------
The module is structured around two classes:

  _ProgressTrackingOptimiser:
    A thin subclass of 'MeshOptimiser' that intercepts the per-iteration
    callback to write a normalised progress value into a shared list. This
    allows the render thread to read the current progress fraction and update
    the ImGui progress bar without any locks or queues.

  InteractiveMeshOptimiser:
    The main application class. Owns the Polyscope scene, all ImGui state
    variables, the background optimisation thread, and the export methods.
    Its 'ui_callback()' method is registered as the Polyscope per-frame
    callback and is invoked on the render thread once per frame.

Threading model
---------------
The optimiser runs on a dedicated background daemon thread launched by
'run_optimisation()'. Communication back to the render thread is through
two shared variables: '_progress' (a one-element list containing a float)
and '_opt_result_ready' (a bool). Both are written by the worker thread and
read by the render thread inside 'ui_callback()'. Because Python's GIL
serialises float and bool assignments, no explicit locking is required.

The render thread never blocks on the worker thread. When '_opt_result_ready'
is True, 'ui_callback()' calls 'update_visuals()' to push the new vertex
positions to the GPU and refresh the planarity colour map, then resets the
flag.

Viewer layout
-------------
The scene shows two meshes simultaneously:
  Left: Original Design -- the mesh as loaded, offset in the negative
        X direction by 60% of the mesh's X extent plus a fixed 0.3-unit gap.
  Right: Working Mesh -- the mesh being optimised, offset in the positive
         X direction by the same amount.

Both meshes display a per-face "Panel Flatness" scalar quantity colour-mapped
with the 'turbo' colour map, normalised to the same range so that the two
meshes are directly visually comparable.

Dependencies
------------
  polyscope -- 3D viewer and ImGui host.
  polyscope.imgui -- ImGui bindings for the control panel.
  tkinter -- Optional; used for a native file-picker fallback on
             platforms where the subprocess-based picker is unavailable.

Entry point
-----------
Run from the project root::

    python src/visualisation/interactive_optimisation.py [mesh.obj]

If no argument is supplied, the viewer loads the default cylinder mesh
from 'data/input/generated/cylinder_10x8.obj' if it exists, or opens a
file picker dialogue.

Author: Muhammad Nabil
Date: March 2026
"""

import copy
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

_HAVE_TK: bool = False
try:
    import tkinter as tk
    from tkinter import filedialog as _fd

    _HAVE_TK = True
except ImportError:
    pass

from src.core.mesh import QuadMesh
from src.io.obj_handler import load_obj, save_obj
from src.io.panel_exporter import export_dxf, export_svg, unfold_mesh
from src.optimisation.energy_terms import compute_planarity_per_face
from src.optimisation.optimiser import (
    MeshOptimiser,
    OptimisationConfig,
    OptimisationResult,
)


class _ProgressTrackingOptimiser(MeshOptimiser):
    """
    MeshOptimiser subclass that exposes real-time iteration progress.

    Overrides the per-iteration callback to write the current optimisation
    progress as a normalised fraction in the range [0, 1) into a shared
    one-element list. The render thread reads this value each frame to
    update the ImGui progress bar without any explicit synchronisation.

    Parameters
    ----------
    config: OptimisationConfig
        Optimisation configuration to pass to the parent 'MeshOptimiser'.
    progress_ref: list of float
        A one-element list whose single entry is updated with the progress
        fraction at each iteration. Must be created by the caller and shared
        with the render thread.
    max_iter: int
        Maximum number of iterations expected. Used to normalise the
        iteration count to the [0, 1) range.
    """

    def __init__(
        self,
        config: OptimisationConfig,
        progress_ref: list,
        max_iter: int,
    ) -> None:
        super().__init__(config)
        self._progress_ref = progress_ref
        self._max_iter = max(max_iter, 1)

    def _create_callback(self, mesh: QuadMesh):  # type: ignore[override]
        """
        Build a per-iteration callback that writes progress and then
        delegates to the parent callback.

        The progress fraction is capped at 0.99 so that the bar only
        reaches 1.0 when the result is confirmed complete, preventing
        premature "done" display while the optimiser is still running.

        Parameters
        ----------
        mesh : QuadMesh
            The mesh being optimised (passed through to parent).

        Returns
        -------
        callable
            A callback function accepting the current parameter vector 'xk'.
        """
        parent_cb = super()._create_callback(mesh)

        def callback(xk):
            parent_cb(xk)
            self._progress_ref[0] = min(self._iteration_count / self._max_iter, 0.99)

        return callback


class InteractiveMeshOptimiser:
    """
    Interactive PQ mesh optimisation application with a Polyscope 3D viewer.

    Loads a quad mesh from an OBJ file, registers it with the Polyscope
    viewer, and provides an ImGui control panel for adjusting optimisation
    weights, running the optimiser, inspecting results, and exporting to
    OBJ, DXF, and SVG formats.

    Parameters
    ----------
    mesh_path : pathlib.Path
        Path to the OBJ mesh file to load on startup.

    Attributes
    ----------
    mesh_path : Path
        Path of the currently loaded mesh file.
    mesh_original : QuadMesh
        The mesh as loaded from the disk, never modified during a session.
    mesh_current : QuadMesh
        The working copy of the mesh, modified by optimisation and
        resettable to 'mesh_original'.
    result : OptimisationResult or None
        The result of the most recently completed optimisation, or None
        if no optimisation has been run yet in the current session.
    """

    def __init__(self, mesh_path: Path):
        self.mesh_path = mesh_path
        self.mesh_original = load_obj(str(mesh_path))
        self.mesh_current = copy.deepcopy(self.mesh_original)

        # Optimisation parameters
        self.wp = [10.0]  # Planarity weight
        self.wf = [1.0]  # Fairness weight
        self.wc = [5.0]  # Closeness weight
        self.wa = [0.0]  # Angle Balance weight
        self.max_iter = [100]

        # State
        self.optimising = [False]
        self.result: Optional[OptimisationResult] = None
        self.show_comparison = [True]
        self.last_saved_path = None

        # Threading / progress
        self._progress: List[float] = [0.0]
        self._opt_thread: Optional[threading.Thread] = None
        self._opt_result_ready: bool = False
        self._load_error: str = ""
        self._export_status: str = ""

        # Cached planarity (updated in update_visuals to avoid per-frame recompute)
        self._planarity_orig: Optional[np.ndarray] = None
        self._planarity_curr: Optional[np.ndarray] = None

        # Cached selection (populated when user clicks; cleared from polyscope
        # via reset_selection so the built-in inspector panel stays hidden)
        self._sel_struct_name: str = ""
        self._sel_elem_type: str = ""
        self._sel_elem_idx: int = -1
        self._sel_position: Optional[np.ndarray] = None

        # Polyscope setup
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_program_name(f"Interactive Optimiser: {mesh_path.name}")

        # Compute side-by-side offset from actual mesh extents
        x_extent = (
            self.mesh_original.vertices[:, 0].max()
            - self.mesh_original.vertices[:, 0].min()
        )
        self._side_offset = x_extent * 0.6 + 0.3  # 60% of width + fixed gap

        # Register meshes
        self.ps_original = ps.register_surface_mesh(
            "Original Design",
            self.mesh_original.vertices
            - np.array([self._side_offset, 0, 0]),  # Offset left
            self.mesh_original.faces,
            smooth_shade=False,
        )
        self.ps_original.set_edge_width(1.5)

        self.ps_current = ps.register_surface_mesh(
            "Working Mesh",
            self.mesh_current.vertices
            + np.array([self._side_offset, 0, 0]),  # Offset right
            self.mesh_current.faces,
            smooth_shade=False,
        )
        self.ps_current.set_edge_width(1.5)

        self.update_visuals()

    def update_visuals(self):
        """
        Refresh the per-face planarity colour maps in the viewer.

        Computes the planarity deviation for every face in both the original
        and current meshes, normalises both colour maps to the same maximum
        value so that the two meshes are directly visually comparable, and
        updates the Polyscope scalar quantities. Called after construction
        and after each completed optimisation.
        """
        planarity_orig = compute_planarity_per_face(self.mesh_original)
        planarity_curr = compute_planarity_per_face(self.mesh_current)
        self._planarity_orig = planarity_orig
        self._planarity_curr = planarity_curr

        vmax = max(planarity_orig.max(), planarity_curr.max(), 1e-10)

        self.ps_original.add_scalar_quantity(
            "Panel Flatness",
            planarity_orig,
            defined_on="faces",
            cmap="turbo",
            vminmax=(0, vmax),
            enabled=True,
        )

        self.ps_current.add_scalar_quantity(
            "Panel Flatness",
            planarity_curr,
            defined_on="faces",
            cmap="turbo",
            vminmax=(0, vmax),
            enabled=True,
        )

    def _optimise_worker(self) -> None:
        """
        Background thread target: run the optimiser and signal completion.

        Constructs an 'OptimisationConfig' from the current UI weight values,
        creates a '_ProgressTrackingOptimiser', runs it on 'mesh_current',
        and stores the result. Sets '_opt_result_ready = True' when complete
        so that the render thread can apply the new vertex positions.
        Always resets 'optimising[0] = False' in a final block, ensuring
        the UI buttons are re-enabled even if the optimiser raises an
        exception.
        """
        try:
            config = OptimisationConfig(
                weights={
                    "planarity": self.wp[0],
                    "fairness": self.wf[0],
                    "closeness": self.wc[0],
                    "angle_balance": self.wa[0],
                },
                max_iterations=self.max_iter[0],
                verbose=True,
            )
            optimiser = _ProgressTrackingOptimiser(
                config, self._progress, self.max_iter[0]
            )
            self.result = optimiser.optimise(self.mesh_current)
            self._progress[0] = 1.0
            self._opt_result_ready = True
        finally:
            self.optimising[0] = False

    def run_optimisation(self) -> None:
        """
        Launch the optimiser on a background daemon thread.

        Does nothing if an optimisation is already in progress. Resets
        the progress counter and the export status string, then starts the
        worker thread. Returns immediately; the render thread polls for
        completion via '_opt_result_ready'.
        """
        if self.optimising[0]:
            return

        print("\n" + "=" * 70)
        print("STARTING OPTIMISATION")
        print("=" * 70)
        print(
            f"Priority settings — "
            f"Flatness: {self.wp[0]:.1f}, "
            f"Smoothness: {self.wf[0]:.1f}, "
            f"Shape fidelity: {self.wc[0]:.1f}, "
            f"Corner balance: {self.wa[0]:.1f}"
        )
        print(
            f"The optimiser will run up to {self.max_iter[0]} improvement steps.\n"
            f"Progress will be printed every 10 steps below."
        )

        self._progress[0] = 0.0
        self.optimising[0] = True
        self._opt_result_ready = False
        self._export_status = ""

        self._opt_thread = threading.Thread(target=self._optimise_worker, daemon=True)
        self._opt_thread.start()

    def _copy_to_clipboard(self, text: str) -> None:
        """
        Copy a text string to the system clipboard.

        Uses the native clipboard utility for the current platform:
          macOS: pbcopy
          Linux: xclip (primary), xsel (fallback)
          Windows: clip (UTF-16 encoding required)

        If no clipboard utility is available, the text is printed to
        standard output instead. Errors are caught and reported non-fatally
        so that a missing clipboard utility does not crash the viewer.

        Parameters
        ----------
        text : str
            The text to copy to the clipboard.
        """
        import platform

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.run(
                    "pbcopy",
                    input=text.encode(),
                    check=True,
                )
                print("Copied to clipboard.")
            elif system == "Linux":
                # Try xclip first (most common on GNOME/Ubuntu desktops)
                try:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=text.encode(),
                        check=True,
                    )
                    print("Copied to clipboard.")
                except FileNotFoundError:
                    # Fallback to xsel (common on KDE and lighter desktops)
                    try:
                        subprocess.run(
                            ["xsel", "--clipboard", "--input"],
                            input=text.encode(),
                            check=True,
                        )
                        print("Copied to clipboard.")
                    except FileNotFoundError:
                        raise RuntimeError(
                            "No clipboard utility found. "
                            "Install xclip (sudo apt install xclip) "
                            "or xsel (sudo apt install xsel)."
                        )
            elif system == "Windows":
                subprocess.run(
                    "clip",
                    input=text.encode("utf-16"),  # clip.exe expects UTF-16 on Windows
                    check=True,
                )
                print("Copied to clipboard.")
            else:
                raise RuntimeError(f"Unsupported platform for clipboard: {system!r}")
        except Exception as exc:
            print(f"Clipboard unavailable ({exc}); printing instead:\n{text}")

    def _build_results_text(self, detailed: bool = False) -> str:
        """
        Build a plain-text summary of the most recent optimisation result.

        Used by the "Copy Results" and "Copy Detailed Breakdown" buttons
        to produce text suitable for pasting into a report or log file.

        Parameters
        ----------
        detailed : bool, optional
            If True, include per-component energy breakdowns, panel flatness
            thresholds, vertex displacement statistics, and solver statistics.
            If False, include only the top-level outcome and key metrics.
            Default is False.

        Returns
        -------
        str
            Multi-line plain-text summary. Returns 'No results yet.' if no
            optimisation has been completed in the current session.
        """
        if not self.result:
            return "No results yet."

        planarity_curr = compute_planarity_per_face(self.mesh_current)
        pct = self.result.energy_reduction_percentage()
        max_p = planarity_curr.max()
        mean_p = planarity_curr.mean()
        n_faces = len(planarity_curr)
        n_pass = int(np.sum(planarity_curr < 1e-3))

        lines = [
            "PQ Mesh Optimisation — Results",
            f"Mesh: {self.mesh_path.name}",
            f"Vertices: {self.mesh_original.n_vertices}   Faces: {self.mesh_original.n_faces}",
            "-" * 50,
        ]

        if self.result.success and n_pass == n_faces:
            lines.append(f"Outcome: All {n_faces} panels are now flat and buildable!")
        elif self.result.success:
            lines.append(
                f"Outcome: {n_pass}/{n_faces} panels flat ({100*n_pass/n_faces:.0f}%)"
            )
        else:
            lines.append(
                f"Outcome: Partial — {n_pass}/{n_faces} panels flat ({100*n_pass/n_faces:.0f}%)"
            )

        flat_desc = (
            "dramatic"
            if pct >= 50
            else "good" if pct >= 20 else "moderate" if pct >= 5 else "small"
        )
        lines.append(
            f"Panel quality improved by: {pct:.0f}%  ({flat_desc} improvement)"
        )

        worst_desc = (
            "within fabrication tolerance"
            if max_p < 1e-3
            else (
                "minor twist remaining"
                if max_p < 1e-2
                else (
                    "some panels still slightly uneven"
                    if max_p < 0.1
                    else "panels still need more optimisation"
                )
            )
        )
        lines.append(f"Worst panel twist: {max_p:.4f}  ({worst_desc})")
        lines.append(f"Average panel twist: {mean_p:.4f}")
        lines.append("(0.000 = perfectly flat; below 0.001 = buildable)")
        lines.append(
            f"Finished in: {self.result.n_iterations} steps, {self.result.execution_time:.2f} s"
        )
        lines.append(
            f"Weights used: planarity={self.wp[0]:.1f}  fairness={self.wf[0]:.1f}  closeness={self.wc[0]:.1f}"
        )

        if detailed:
            lines += [
                "",
                "--- Score Breakdown ---",
                f"Score before: {self.result.initial_energy:.4f}",
                f"Score after:  {self.result.final_energy:.4f}",
                f"Total reduction: {self.result.initial_energy - self.result.final_energy:.4f}  ({pct:.0f}% better)",
            ]
            if (
                self.result.component_energies_initial
                and self.result.component_energies_final
            ):
                _labels = {
                    "planarity": "Flat panels",
                    "fairness": "Smooth surface",
                    "closeness": "Keep shape",
                }
                for key in self.result.component_energies_initial:
                    e0 = self.result.component_energies_initial[key]
                    ef = self.result.component_energies_final[key]
                    label = _labels.get(key, key)
                    pct_c = (e0 - ef) / e0 * 100.0 if e0 > 0 else 0.0
                    lines.append(
                        f"  {label}: {e0:.3f} -> {ef:.3f}  ({pct_c:.0f}% better)"
                    )

            lines += [
                "",
                "--- Panel Flatness Detail ---",
            ]
            planarity_orig = compute_planarity_per_face(self.mesh_original)
            for label, thresh in [
                ("Buildable (<0.001)", 1e-3),
                ("Near-flat (<0.010)", 1e-2),
                ("Acceptable (<0.100)", 0.1),
            ]:
                n_orig = int(np.sum(planarity_orig < thresh))
                n_curr = int(np.sum(planarity_curr < thresh))
                lines.append(
                    f"  {label}: Before {n_orig}/{n_faces}  ->  After {n_curr}/{n_faces}"
                )

            lines += [
                "",
                "--- Shape Change ---",
            ]
            displacements = np.linalg.norm(
                self.mesh_current.vertices - self.mesh_original.vertices, axis=1
            )
            bbox_diag = np.linalg.norm(
                self.mesh_original.vertices.max(axis=0)
                - self.mesh_original.vertices.min(axis=0)
            )
            lines.append(f"  Biggest single move: {displacements.max():.4f}")
            lines.append(f"  Average move:        {displacements.mean():.4f}")
            if bbox_diag > 0:
                rel_pct = displacements.max() / bbox_diag * 100
                lines.append(f"  Biggest move as % of mesh size: {rel_pct:.1f}%")
            lines.append(
                f"  Corners that moved: {int(np.sum(displacements > 1e-8))} of {len(displacements)}"
            )

            lines += [
                "",
                "--- Solver Stats ---",
                f"  Steps taken:          {self.result.n_iterations}",
                f"  Quality checks run:   {self.result.n_function_evaluations}",
                f"  Direction checks run: {self.result.n_gradient_evaluations}",
                f"  Total time:           {self.result.execution_time:.2f} s",
            ]

        return "\n".join(lines)

    def reset_mesh(self):
        """
        Reset the working mesh to the original loaded geometry.

        Replaces 'mesh_current' with a deep copy of 'mesh_original',
        updates the Polyscope vertex positions, refreshes the planarity
        colour map, and clears the stored result and export status. Does
        not reload from disk; uses the in-memory copy of the original.
        """
        self.mesh_current = copy.deepcopy(self.mesh_original)
        verts_offset = self.mesh_current.vertices + np.array([self._side_offset, 0, 0])
        self.ps_current.update_vertex_positions(verts_offset)
        self.update_visuals()
        self.result = None
        self._export_status = ""
        print("Reset to original mesh")

    def load_new_mesh(self, path: Path) -> None:
        """
        Load a new OBJ mesh file and refresh the viewer.

        Replaces both 'mesh_original' and 'mesh_current' with the new mesh,
        re-registers both Polyscope surface meshes with updated geometry and
        a recalculated side-by-side offset, and refreshes the planarity
        colour maps. Clears the stored result and export status.

        On failure (for example, if the OBJ file cannot be parsed),
        stores the error message in '_load_error' for display in the UI
        rather than raising an exception.

        Parameters
        ----------
        path : pathlib.Path
            Path to the new OBJ file to load.
        """
        try:
            new_mesh = load_obj(str(path))
        except Exception as exc:
            self._load_error = str(exc)
            return

        self._load_error = ""
        self.mesh_path = path
        self.mesh_original = new_mesh
        self.mesh_current = copy.deepcopy(new_mesh)
        self.result = None
        self._opt_result_ready = False
        self._export_status = ""

        x_extent = (
            self.mesh_original.vertices[:, 0].max()
            - self.mesh_original.vertices[:, 0].min()
        )
        self._side_offset = x_extent * 0.6 + 0.3

        self.ps_original = ps.register_surface_mesh(
            "Original Design",
            self.mesh_original.vertices - np.array([self._side_offset, 0, 0]),
            self.mesh_original.faces,
            smooth_shade=False,
        )
        self.ps_original.set_edge_width(1.5)

        self.ps_current = ps.register_surface_mesh(
            "Working Mesh",
            self.mesh_current.vertices + np.array([self._side_offset, 0, 0]),
            self.mesh_current.faces,
            smooth_shade=False,
        )
        self.ps_current.set_edge_width(1.5)

        self.update_visuals()
        print(f"Loaded: {path}")

    @staticmethod
    def _pick_file() -> Optional[str]:
        """
        Open a native file-picker dialogue using a subprocess call.

        Invokes the platform-appropriate native file selection dialogue
        without using Tkinter, which conflicts with the GLFW event loop
        used by Polyscope:
          macOS: osascript (AppleScript choose a file)
          Linux: zenity (primary), kdialog (fallback)
          Windows: PowerShell OpenFileDialog

        Each attempt times out after 120 seconds. If the platform is not
        recognised or all tools are unavailable, it returns None without
        raising an exception.

        Returns
        -------
        str or None
            Absolute path of the selected file as a string, or None if
            the user cancelled or no file-picker was available.
        """
        import platform

        system = platform.system()

        if system == "Darwin":
            script = (
                'POSIX path of (choose file of type {"obj"} '
                'with prompt "Select OBJ mesh file")'
            )
            try:
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass

        elif system == "Linux":
            # Try zenity (GTK / GNOME)
            try:
                result = subprocess.run(
                    [
                        "zenity",
                        "--file-selection",
                        "--title=Select OBJ mesh file",
                        "--file-filter=OBJ files (*.obj) | *.obj",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except FileNotFoundError:
                pass
            # Fallback: kdialog (KDE)
            try:
                result = subprocess.run(
                    ["kdialog", "--getopenfilename", ".", "*.obj"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except FileNotFoundError:
                pass

        elif system == "Windows":
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$d = New-Object System.Windows.Forms.OpenFileDialog;"
                "$d.Title = 'Select OBJ mesh file';"
                "$d.Filter = 'OBJ files (*.obj)|*.obj|All files (*.*)|*.*';"
                "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.FileName }"
            )
            try:
                result = subprocess.run(
                    ["powershell", "-NonInteractive", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                pass

        return None

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _export_obj(self) -> None:
        """
        Export the current working mesh as an OBJ file.

        Saves to 'data/output/optimised_meshes/' relative to the project
        root, using the pattern '<original_stem>_optimised_<timestamp>.obj'.
        Updates '_export_status' with the result for display in the UI.
        """
        try:
            output_dir = Path(__file__).parents[2] / "data/output/optimised_meshes"
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{self.mesh_path.stem}_optimised_{self._timestamp()}.obj"
            out_path = output_dir / filename
            save_obj(self.mesh_current, str(out_path))
            self._export_status = f"Saved OBJ: {out_path.name}"
            print(f"Exported OBJ: {out_path}")
        except Exception as exc:
            self._export_status = f"OBJ export failed: {exc}"

    def _export_dxf(self) -> None:
        """
        Unfold the current working mesh and export the panels as a DXF file.

        Saves to 'data/output/panels/' relative to the project root, using
        the pattern '<original_stem>_panels_<timestamp>.dxf'. Updates
        '_export_status' with the result for display in the UI.
        """
        try:
            panels, _ = unfold_mesh(self.mesh_current)
            output_dir = Path(__file__).parents[2] / "data/output/panels"
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{self.mesh_path.stem}_panels_{self._timestamp()}.dxf"
            out_path = output_dir / filename
            export_dxf(panels, str(out_path))
            self._export_status = f"Saved DXF: {out_path.name}"
            print(f"Exported DXF: {out_path}")
        except Exception as exc:
            self._export_status = f"DXF export failed: {exc}"

    def _export_svg(self) -> None:
        """
        Unfold the current working mesh and export the panels as an SVG file.

        Saves to 'data/output/panels/' relative to the project root, using
        the pattern '<original_stem>_panels_<timestamp>.svg'. Updates
        '_export_status' with the result for display in the UI.
        """
        try:
            panels, _ = unfold_mesh(self.mesh_current)
            output_dir = Path(__file__).parents[2] / "data/output/panels"
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{self.mesh_path.stem}_panels_{self._timestamp()}.svg"
            out_path = output_dir / filename
            export_svg(panels, str(out_path))
            self._export_status = f"Saved SVG: {out_path.name}"
            print(f"Exported SVG: {out_path}")
        except Exception as exc:
            self._export_status = f"SVG export failed: {exc}"

    def ui_callback(self):
        """
        Per-frame ImGui callback registered with the Polyscope viewer.

        Called once per render frame by the Polyscope event loop on the
        render thread. Responsible for:
          1. Detecting and applying completed optimisation results from the
             background thread.
          2. Reading and caching the current Polyscope selection, then
             clearing it to suppress the built-in inspector panel.
          3. Rendering all ImGui widgets: the file browser, mesh info,
             weight sliders, action buttons, selection inspector, progress
             bar, result display, and export buttons.

        All UI states are maintained as instance variables on the class.
        No external locks are required because the render thread is the
        only writer of UI state, and shared communication with the worker
        thread uses single-assignment bool and float variables protected
        by the GIL.
        """
        # -----------------------------------------------------------------
        # Pending-result handler — runs on the render thread
        # -----------------------------------------------------------------
        if self._opt_result_ready:
            verts_offset = self.mesh_current.vertices + np.array(
                [self._side_offset, 0, 0]
            )
            self.ps_current.update_vertex_positions(verts_offset)
            self.update_visuals()
            if self.result is not None:
                print("\nOptimisation complete")
                print(
                    f"Energy reduced by {self.result.energy_reduction_percentage():.1f}%"
                )
                print("=" * 70 + "\n")
            self._opt_result_ready = False

        # -----------------------------------------------------------------
        # Read & cache any fresh selection, then clear it so polyscope's
        # built-in inspector panel below the main window has nothing to show.
        # -----------------------------------------------------------------
        try:
            if ps.have_selection():
                result = ps.get_selection()
                if result.is_hit and result.structure_type_name == "Surface Mesh":
                    self._sel_struct_name = result.structure_name
                    self._sel_elem_type = result.structure_data.get("element_type", "")
                    self._sel_elem_idx = result.structure_data.get(
                        "index", result.local_index
                    )
                    self._sel_position = result.position
            ps.reset_selection()
        except Exception:
            pass

        psim.PushItemWidth(200)

        # -----------------------------------------------------------------
        # Header
        # -----------------------------------------------------------------
        psim.TextUnformatted("PQ Mesh Optimiser")
        psim.Separator()

        # -----------------------------------------------------------------
        # File browser
        # -----------------------------------------------------------------
        psim.TextUnformatted("Load Mesh")
        psim.TextUnformatted(f"  {self.mesh_path}")
        if psim.Button("Browse Mesh File"):
            chosen = self._pick_file()
            if chosen:
                self.load_new_mesh(Path(chosen))
        if self._load_error:
            psim.TextUnformatted(f"  Error loading file: {self._load_error}")
        psim.Separator()

        # -----------------------------------------------------------------
        # Mesh info
        # -----------------------------------------------------------------
        psim.TextUnformatted(f"File: {self.mesh_path.name}")
        psim.TextUnformatted(f"Corner points: {self.mesh_original.n_vertices}")
        psim.TextUnformatted(f"Panels: {self.mesh_original.n_faces}")
        psim.Separator()

        # -----------------------------------------------------------------
        # Heat map guidance
        # -----------------------------------------------------------------
        psim.TextUnformatted("Heat Map Guidance:")
        psim.TextUnformatted(
            "Cool colour = even panels         Warm colour = uneven panels"
        )
        psim.Separator()

        # -----------------------------------------------------------------
        # Weight controls
        # -----------------------------------------------------------------
        psim.TextUnformatted("Optimisation Weights:")
        psim.TextUnformatted("  (Higher value = stronger priority for that goal)")

        _, self.wp[0] = psim.SliderFloat("Planarity", self.wp[0], 0.1, 100.0)
        psim.TextUnformatted("  How hard to push each quad face to be flat.")
        psim.TextUnformatted("  Increase this for fabrication-ready flat panels.")
        psim.Separator()

        _, self.wf[0] = psim.SliderFloat("Fairness", self.wf[0], 0.1, 20.0)
        psim.TextUnformatted("  How smooth and even the surface should be.")
        psim.TextUnformatted("  Increase this to reduce bumps and irregular patches.")
        psim.Separator()

        _, self.wc[0] = psim.SliderFloat("Closeness", self.wc[0], 0.0, 50.0)
        psim.TextUnformatted(
            "  How close the result should stay to the original shape."
        )
        psim.TextUnformatted("  Increase this to preserve the original design intent.")
        psim.Separator()

        _, self.wa[0] = psim.SliderFloat("Angle Balance", self.wa[0], 0.0, 20.0)
        psim.TextUnformatted(
            "  Controls how evenly corners meet at each joining point."
        )
        psim.TextUnformatted(
            "  Increase this if panels need to connect neatly in real life,"
        )
        psim.TextUnformatted(
            "  such as when building an actual structure from the panels."
        )
        psim.Separator()

        _, self.max_iter[0] = psim.SliderInt(
            "Max Iterations", self.max_iter[0], 10, 500
        )
        psim.TextUnformatted("  Maximum number of improvement steps to run.")
        psim.TextUnformatted(
            "  More iterations = potentially better result, but slower."
        )

        # -----------------------------------------------------------------
        # Action buttons (Optimise disabled while running)
        # -----------------------------------------------------------------
        # Snapshot state once so Begin/EndDisabled are always paired
        try:
            _aw = psim.GetContentRegionAvail()[0]
            _off = (_aw - 130) * 0.5
            if _off > 0:
                psim.SetCursorPosX(psim.GetCursorPosX() + _off)
        except Exception:
            pass
        _running = self.optimising[0]
        if _running:
            psim.BeginDisabled(True)
        if psim.Button("Optimise"):
            self.run_optimisation()
        if _running:
            psim.EndDisabled()

        psim.SameLine()
        if psim.Button("Reset"):
            self.reset_mesh()

        # -----------------------------------------------------------------
        # Selection Info
        # -----------------------------------------------------------------
        psim.Separator()
        psim.TextUnformatted("Selection Info:")
        if self._sel_struct_name:
            if self._sel_struct_name == "Original Design":
                mesh_ref = self.mesh_original
                planarity_ref = self._planarity_orig
            elif self._sel_struct_name == "Working Mesh":
                mesh_ref = self.mesh_current
                planarity_ref = self._planarity_curr
            else:
                mesh_ref = None
                planarity_ref = None

            psim.BulletText(f"Mesh: {self._sel_struct_name}")
            elem_type = self._sel_elem_type.lower() if self._sel_elem_type else ""
            psim.BulletText(
                f"Type: {elem_type or 'unknown'}   Index: #{self._sel_elem_idx}"
            )

            if mesh_ref is not None:
                idx = self._sel_elem_idx
                if elem_type == "vertex" and 0 <= idx < len(mesh_ref.vertices):
                    v = mesh_ref.vertices[idx]
                    psim.BulletText(f"X: {v[0]:.4f}")
                    psim.BulletText(f"Y: {v[1]:.4f}")
                    psim.BulletText(f"Z: {v[2]:.4f}")
                elif elem_type == "face" and 0 <= idx < len(mesh_ref.faces):
                    face = mesh_ref.faces[idx]
                    psim.BulletText(f"Corner indices: {list(face)}")
                    if planarity_ref is not None and idx < len(planarity_ref):
                        p_val = planarity_ref[idx]
                        # Always fetch the original planarity for comparison
                        p_orig = (
                            self._planarity_orig[idx]
                            if self._planarity_orig is not None
                            and idx < len(self._planarity_orig)
                            else None
                        )
                        psim.BulletText(f"Panel uneven (Selected):      {p_val:.6f}")
                        if p_orig is not None:
                            psim.BulletText(f"Panel uneven (original): {p_orig:.6f}")
                            if p_orig > 1e-12:
                                delta = p_orig - p_val
                                pct = delta / p_orig * 100.0
                                if pct >= 0:
                                    psim.BulletText(
                                        f"Improvement: -{delta:.6f}  ({pct:.1f}% flatter)"
                                    )
                                else:
                                    psim.BulletText(
                                        f"Change: +{abs(delta):.6f}  ({abs(pct):.1f}% worse)"
                                    )
                            else:
                                psim.BulletText(
                                    "Improvement: N/A (original was already flat)"
                                )
                        if p_val < 1e-3:
                            psim.TextUnformatted("  Status: Buildable (flat)")
                        elif p_val < 1e-2:
                            psim.TextUnformatted("  Status: Near-flat")
                        elif p_val < 0.1:
                            psim.TextUnformatted("  Status: Slightly uneven")
                        else:
                            psim.TextUnformatted("  Status: Needs more optimisation")
        else:
            psim.TextUnformatted("  Click a panel or corner point to inspect it.")

        # -----------------------------------------------------------------
        # Progress bar
        # -----------------------------------------------------------------
        if self.optimising[0]:
            psim.ProgressBar(self._progress[0])
            psim.TextUnformatted(f"  Optimising... {self._progress[0] * 100:.0f}%")

        psim.Separator()

        # -----------------------------------------------------------------
        # Results display (only after optimisation completes)
        # -----------------------------------------------------------------
        if self.result:
            planarity_curr = compute_planarity_per_face(self.mesh_current)
            pct = self.result.energy_reduction_percentage()
            max_p = planarity_curr.max()
            mean_p = planarity_curr.mean()
            n_faces = len(planarity_curr)
            n_pass = int(np.sum(planarity_curr < 1e-3))

            psim.TextUnformatted(" Results:")

            # Overall verdict
            if self.result.success and n_pass == n_faces:
                psim.BulletText(
                    f"Outcome: All {n_faces} panels are now flat and buildable!"
                )
            elif self.result.success:
                psim.BulletText(
                    f"Outcome: {n_pass}/{n_faces} panels are now flat ({100*n_pass/n_faces:.0f}%)."
                )
            else:
                psim.BulletText(
                    f"Outcome: Partial improvement — {n_pass}/{n_faces} panels flat ({100*n_pass/n_faces:.0f}%)."
                )
                psim.TextUnformatted(
                    "  Try increasing Max Iterations or the Planarity weight."
                )

            # How much flatter?
            if pct >= 50:
                flat_desc = "dramatic improvement"
            elif pct >= 20:
                flat_desc = "good improvement"
            elif pct >= 5:
                flat_desc = "moderate improvement"
            else:
                flat_desc = "small improvement — try higher Planarity weight"
            psim.BulletText(f"Panel quality improved by {pct:.0f}%  ({flat_desc})")

            # Worst remaining panel
            if max_p < 1e-3:
                worst_desc = "excellent — within fabrication tolerance"
            elif max_p < 1e-2:
                worst_desc = "good — minor twist remaining"
            elif max_p < 0.1:
                worst_desc = "noticeable — some panels still slightly uneven"
            else:
                worst_desc = "high — panels still need more optimisation"
            psim.BulletText(f"Worst panel twist: {max_p:.4f}  ({worst_desc})")
            psim.BulletText(
                f"Average panel twist: {mean_p:.4f}  (across all {n_faces} panels)"
            )
            psim.TextUnformatted(
                "  (0.000 = perfectly flat; anything below 0.001 is buildable)"
            )

            # Speed
            psim.BulletText(
                f"Finished in {self.result.n_iterations} steps, {self.result.execution_time:.2f} seconds"
            )

            # --- Detailed Information (collapsible) ---
            if psim.TreeNode("Detailed Breakdown"):

                # Energy breakdown
                if psim.TreeNode("Score Breakdown"):
                    psim.TextUnformatted("The optimiser minimises a combined 'score'.")
                    psim.TextUnformatted(
                        "Lower score = flatter panels, smoother surface, shape better preserved."
                    )
                    psim.Separator()
                    psim.BulletText(f"Score before: {self.result.initial_energy:.3f}")
                    psim.BulletText(f"Score after:  {self.result.final_energy:.3f}")
                    abs_red = self.result.initial_energy - self.result.final_energy
                    psim.BulletText(
                        f"Total reduction: {abs_red:.3f}  ({pct:.0f}% better)"
                    )
                    if (
                        self.result.component_energies_initial
                        and self.result.component_energies_final
                    ):
                        psim.Separator()
                        psim.TextUnformatted("Per-goal contribution:")
                        _labels = {
                            "planarity": "Flat panels   ",
                            "fairness": "Smooth surface",
                            "closeness": "Keep shape    ",
                            "angle_balance": "Angle Balance ",
                        }
                        for key in self.result.component_energies_initial:
                            e0 = self.result.component_energies_initial[key]
                            ef = self.result.component_energies_final[key]
                            label = _labels.get(key, key.capitalize())
                            if e0 > 0:
                                pct_c = (e0 - ef) / e0 * 100.0
                                psim.BulletText(
                                    f"{label}  {e0:.3f} -> {ef:.3f}  ({pct_c:.0f}% better)"
                                )
                            else:
                                psim.BulletText(f"{label}  {e0:.3f} -> {ef:.3f}")
                    psim.TreePop()

                # Solver statistics
                if psim.TreeNode("How Hard Did It Work?"):
                    psim.TextUnformatted(
                        "These numbers show the computational effort used."
                    )
                    psim.Separator()
                    psim.BulletText(f"Steps taken: {self.result.n_iterations}")
                    psim.TextUnformatted(
                        "  Each step nudges all vertices to reduce the score."
                    )
                    psim.BulletText(
                        f"Quality checks run: {self.result.n_function_evaluations}"
                    )
                    psim.TextUnformatted(
                        "  Each check measures the score of the whole mesh."
                    )
                    psim.BulletText(
                        f"Direction checks run: {self.result.n_gradient_evaluations}"
                    )
                    psim.TextUnformatted(
                        "  Each check figures out which way to nudge next."
                    )
                    psim.BulletText(f"Total time: {self.result.execution_time:.2f} s")
                    if self.result.n_iterations > 0:
                        ms_per = (
                            self.result.execution_time / self.result.n_iterations * 1000
                        )
                        psim.BulletText(f"Time per step: {ms_per:.1f} ms")
                    psim.TreePop()

                # Planarity quality
                if psim.TreeNode("Panel Flatness Detail"):
                    psim.TextUnformatted("How many panels meet each quality level?")
                    psim.TextUnformatted(
                        "Fabrication tolerance = twist < 0.001 (real-world buildable)."
                    )
                    psim.Separator()
                    planarity_orig = compute_planarity_per_face(self.mesh_original)
                    thresholds = [
                        ("Buildable   (twist < 0.001)", 1e-3),
                        ("Near-flat   (twist < 0.010)", 1e-2),
                        ("Acceptable  (twist < 0.100)", 0.1),
                    ]
                    for thresh_label, thresh in thresholds:
                        n_orig = int(np.sum(planarity_orig < thresh))
                        n_curr = int(np.sum(planarity_curr < thresh))
                        psim.BulletText(f"{thresh_label}:")
                        psim.BulletText(
                            f"  Before: {n_orig}/{n_faces} panels  |  After: {n_curr}/{n_faces} panels"
                        )
                    psim.Separator()
                    psim.BulletText(
                        f"Consistency before: {planarity_orig.std():.4f} std dev"
                    )
                    psim.BulletText(
                        f"Consistency after:  {planarity_curr.std():.4f} std dev"
                    )
                    psim.TextUnformatted("  (Lower = panels are more uniformly flat)")
                    psim.TreePop()

                # Vertex displacement
                if psim.TreeNode("How Much Did the Shape Change?"):
                    psim.TextUnformatted("Shows how far each corner point moved from")
                    psim.TextUnformatted(
                        "its original position to achieve flat panels."
                    )
                    psim.TextUnformatted(
                        "Small movement = original design well preserved."
                    )
                    psim.Separator()
                    displacements = np.linalg.norm(
                        self.mesh_current.vertices - self.mesh_original.vertices, axis=1
                    )
                    bbox_diag = np.linalg.norm(
                        self.mesh_original.vertices.max(axis=0)
                        - self.mesh_original.vertices.min(axis=0)
                    )
                    psim.BulletText(f"Biggest single move: {displacements.max():.4f}")
                    psim.BulletText(f"Average move:        {displacements.mean():.4f}")
                    if bbox_diag > 0:
                        rel_pct = displacements.max() / bbox_diag * 100
                        if rel_pct < 1:
                            move_rating = "tiny — shape is virtually unchanged"
                        elif rel_pct < 5:
                            move_rating = "small — shape is mostly preserved"
                        else:
                            move_rating = "noticeable — shape has changed meaningfully"
                        psim.BulletText(
                            f"Biggest move as % of mesh size: {rel_pct:.1f}%  ({move_rating})"
                        )
                    n_moved = int(np.sum(displacements > 1e-8))
                    psim.BulletText(
                        f"Corners that moved: {n_moved} of {len(displacements)}"
                    )
                    psim.TreePop()

                psim.TreePop()  # Detailed Breakdown

            # Copy buttons (centered, no separator above)
            try:
                _aw = psim.GetContentRegionAvail()[0]
                _off = (_aw - 280) * 0.5
                if _off > 0:
                    psim.SetCursorPosX(psim.GetCursorPosX() + _off)
            except Exception:
                pass
            if psim.Button("Copy Results"):
                self._copy_to_clipboard(self._build_results_text(detailed=False))
            psim.SameLine()
            if psim.Button("Copy Detailed Breakdown"):
                self._copy_to_clipboard(self._build_results_text(detailed=True))
            psim.Separator()

            # Export buttons (centered)
            psim.TextUnformatted("Export:")
            try:
                _aw = psim.GetContentRegionAvail()[0]
                _off = (_aw - 260) * 0.5
                if _off > 0:
                    psim.SetCursorPosX(psim.GetCursorPosX() + _off)
            except Exception:
                pass
            if psim.Button("Export OBJ"):
                self._export_obj()
            psim.SameLine()
            if psim.Button("Export DXF"):
                self._export_dxf()
            psim.SameLine()
            if psim.Button("Export SVG"):
                self._export_svg()
            if self._export_status:
                psim.TextUnformatted(f"  {self._export_status}")

        psim.PopItemWidth()

    def run(self):
        """
        Register the UI callback and start the Polyscope viewer event loop.

        This method blocks until the viewer window is closed. All further
        execution happens inside 'ui_callback()' on the render thread.
        """
        ps.set_user_callback(self.ui_callback)
        ps.show()


DEFAULT_MESH = (
    Path(__file__).parent.parent.parent
    / "data"
    / "input"
    / "generated"
    / "cylinder_10x8.obj"
)


def main():
    """
    Entry point for the interactive optimisation application.

    Parses the command-line argument (if provided) to determine the mesh
    file to load. If no argument is given, attempts to use the default
    cylinder mesh; if that does not exist, opens a Tkinter file picker
    (if available) or prints usage instructions and exits.

    Validates that the resolved mesh file exists, prints the startup banner
    and viewer controls, warms up any Numba JIT kernels to avoid an
    unexplained freeze on the first "Optimise" click, then constructs the
    'InteractiveMeshOptimiser' and starts the viewer.
    """
    if len(sys.argv) < 2:
        if DEFAULT_MESH.exists():
            mesh_path = DEFAULT_MESH
            print(f"No mesh specified — using default: {mesh_path}")
        elif _HAVE_TK:
            root = tk.Tk()  # type: ignore[possibly-undefined]
            root.withdraw()
            root.attributes("-topmost", True)
            chosen = _fd.askopenfilename(  # type: ignore[possibly-undefined]
                title="Select OBJ mesh file",
                filetypes=[("OBJ files", "*.obj"), ("All files", "*")],
            )
            root.destroy()
            if not chosen:
                print("No file selected. Exiting.")
                sys.exit(0)
            mesh_path = Path(chosen)
        else:
            print(
                "Usage: python src/visualisation/interactive_optimisation.py <mesh.obj>"
            )
            print("\nExample:")
            print(
                "  python src/visualisation/interactive_optimisation.py"
                " data/input/reference_datasets/spot_quadrangulated.obj"
            )
            sys.exit(1)
    else:
        mesh_path = Path(sys.argv[1])

    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        sys.exit(1)

    print("=" * 70)
    print("INTERACTIVE MESH OPTIMISATION")
    print("=" * 70)
    print("\nControls:")
    print("  - Browse or pass a mesh file as an argument")
    print("  - Adjust sliders to change weights")
    print("  - Click 'Optimise' to run (progress bar shows status)")
    print("  - Click 'Reset' to restore original")
    print("  - Click 'Export OBJ / DXF / SVG' to save results")
    print("\nViewer controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("=" * 70 + "\n")

    # ── Warm up Numba JIT kernels before launching the viewer ─────────────────
    # Numba kernels are compiled with cache=True, but the first call in a fresh
    # environment still incurs a JIT compilation stall of ~2-5 seconds.
    # Warming up here produces clear console feedback rather than an unexplained
    # freeze on the first "Optimise" click inside the viewer.
    try:
        from src.backends import HAS_NUMBA, warmup_numba_kernels

        if HAS_NUMBA:
            print(
                "Warming up Numba kernels (one-time compile, ~2-5 seconds on first run)..."
            )
            warmup_numba_kernels()
            print("Kernels ready.\n")
    except (ImportError, Exception) as _warmup_exc:
        # Non-fatal: the kernels will compile on first use instead
        print(
            f"Note: Numba warmup skipped ({_warmup_exc}). Kernels will compile on first use.\n"
        )

    app = InteractiveMeshOptimiser(mesh_path)
    app.run()


if __name__ == "__main__":
    main()
