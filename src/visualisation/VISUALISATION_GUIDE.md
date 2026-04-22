# src/visualisation — Visualisation Guide

This package provides the primary end-user interface for the project: an interactive 3D mesh viewer built on [Polyscope](https://polyscope.run) with a Dear ImGui control panel that requires no Python programming to operate.

---

## Package contents

| File | Role |
| ---- | ---- |
| `interactive_optimisation.py` | Main application (57 KB). Contains `_ProgressTrackingOptimiser` and `InteractiveMeshOptimiser`. Entry point via `main()`. |
| `__init__.py` | Package marker; exports nothing. |

---

## Architecture

### Threading model

The application runs two concurrent threads:

- **Render thread** — the Polyscope GLFW event loop. Drives ImGui rendering via `ui_callback()` once per frame. This thread is the sole writer of all UI state variables.
- **Worker thread** — a Python `daemon` thread launched by `run_optimisation()`. Runs the L-BFGS-B optimiser. Communicates with the render thread through two shared variables:
  - `_progress: list[float]` — a one-element list updated by the worker with the current progress fraction in `[0, 1)`. Single-element list assignment is GIL-atomic; no lock is required.
  - `_opt_result_ready: bool` — set to `True` by the worker when optimisation is complete. Read by the render thread inside `ui_callback()`, which then calls `update_visuals()` and resets the flag.

Because Python's Global Interpreter Lock serialises `float` and `bool` assignment, no explicit `threading.Lock` is needed for these two shared variables.

### Viewer layout

The Polyscope scene shows two meshes simultaneously, side by side:

| Mesh name | Content | X offset |
| --------- | ------- | -------- |
| `Original Design` | Loaded mesh, never modified | Negative X by `x_extent * 0.6 + 0.3` |
| `Working Mesh` | Mesh being optimised, mutable | Positive X by the same amount |

Both meshes display a per-face `Panel Flatness` scalar quantity colour-mapped with the `turbo` colour map, normalised to the same `vmax = max(planarity_orig.max(), planarity_curr.max())` so that the two meshes are directly visually comparable.

---

## Classes

### `_ProgressTrackingOptimiser`

A thin subclass of `MeshOptimiser` that intercepts the per-iteration callback to write a normalised progress fraction into the shared `_progress` list.

```python
class _ProgressTrackingOptimiser(MeshOptimiser):
    def __init__(self, config, progress_ref: list, max_iter: int): ...
    def _create_callback(self, mesh: QuadMesh): ...
```

**Key detail.** The progress fraction is capped at `0.99` until the worker sets `_opt_result_ready = True`, preventing the progress bar from displaying "100%" while the optimiser is still running.

---

### `InteractiveMeshOptimiser`

The main application class. Owns the Polyscope scene, all ImGui state, the background thread, and the export methods.

#### Construction — `__init__(mesh_path: Path)`

1. Loads the mesh with `load_obj`; stores a deep copy as `mesh_original` and a mutable copy as `mesh_current`.
2. Computes `_side_offset = x_extent * 0.6 + 0.3` from the actual mesh extents.
3. Registers both meshes with Polyscope using `ps.register_surface_mesh`.
4. Calls `update_visuals()` to compute initial planarity colour maps.

**Default weight values** at startup:

| Slider | Variable | Default |
| ------ | -------- | ------- |
| Planarity | `self.wp` | `10.0` |
| Fairness | `self.wf` | `1.0` |
| Closeness | `self.wc` | `5.0` |
| Angle Balance | `self.wa` | `0.0` |
| Max Iterations | `self.max_iter` | `100` |

#### `update_visuals()`

Recomputes `compute_planarity_per_face` for both meshes, determines a shared `vmax`, and pushes updated `Panel Flatness` scalar quantities to the GPU. Called after construction, after each completed optimisation, and after `reset_mesh()`.

#### `run_optimisation()`

Launches `_optimise_worker` on a background daemon thread. Does nothing if `optimising[0]` is already `True`. Resets `_progress[0] = 0.0` and `_export_status = ""` before starting.

#### `_optimise_worker()`

Background thread target. Constructs `OptimisationConfig` from current UI slider values, creates a `_ProgressTrackingOptimiser`, calls `optimiser.optimise(self.mesh_current)`, stores the result, sets `_progress[0] = 1.0`, and sets `_opt_result_ready = True`. Always resets `optimising[0] = False` in a `finally` block even if the optimiser raises.

#### `ui_callback()`

Per-frame ImGui callback registered with Polyscope via `ps.set_user_callback`. Responsibilities in order:

1. **Pending-result handler** — if `_opt_result_ready`, pushes new vertex positions to the GPU via `ps_current.update_vertex_positions`, calls `update_visuals()`, prints the energy reduction percentage, and resets `_opt_result_ready`.
2. **Selection caching** — reads the Polyscope selection (vertex or face), caches `_sel_struct_name`, `_sel_elem_type`, `_sel_elem_idx`, and `_sel_position`, then calls `ps.reset_selection()` to suppress the built-in inspector.
3. **Widget rendering** — renders all ImGui panels: file browser, mesh info, heat-map guidance, weight sliders (planarity, fairness, closeness, angle balance, max iterations), action buttons, selection inspector, progress bar, results display with collapsible `Detailed Breakdown`, copy buttons, and export buttons.

#### `reset_mesh()`

Replaces `mesh_current` with `copy.deepcopy(mesh_original)`, updates vertex positions in Polyscope, refreshes planarity colour maps, and clears `result` and `_export_status`.

#### `load_new_mesh(path: Path)`

Loads a new OBJ file, recomputes `_side_offset` from the new mesh extents, re-registers both Polyscope surface meshes, refreshes colour maps, and resets all state. On `load_obj` failure, stores the error in `_load_error` for display in the UI rather than raising.

#### `_pick_file() -> str | None` (static)

Opens a native file-picker dialogue using platform-specific subprocess calls, avoiding Tkinter which conflicts with the GLFW event loop:

| Platform | Tool used | Fallback |
| -------- | --------- | -------- |
| macOS | `osascript` (AppleScript) | None |
| Linux | `zenity` (GTK/GNOME) | `kdialog` (KDE) |
| Windows | PowerShell `OpenFileDialog` | None |

Each call has a 120-second timeout. Returns `None` without raising if no tool is available.

#### `_copy_to_clipboard(text: str)`

Copies text to the system clipboard using platform-appropriate utilities:

| Platform | Command |
| -------- | ------- |
| macOS | `pbcopy` |
| Linux | `xclip -selection clipboard` or `xsel --clipboard --input` |
| Windows | `clip` (UTF-16 encoded) |

Falls back to printing the text to stdout if no utility is found.

#### `_build_results_text(detailed: bool) -> str`

Builds a plain-text summary of the most recent `OptimisationResult` for clipboard export. In non-detailed mode, includes outcome, energy reduction percentage, worst/mean planarity, iteration count, and weights. In detailed mode, additionally includes per-component energy breakdown, panel flatness thresholds (buildable / near-flat / acceptable), vertex displacement statistics, and solver statistics (function evaluations, gradient evaluations, wall-clock time).

#### Export methods

| Method | Output location | Format | Filename pattern |
| ------ | --------------- | ------ | ---------------- |
| `_export_obj()` | `data/output/optimised_meshes/` | OBJ | `<stem>_optimised_<timestamp>.obj` |
| `_export_dxf()` | `data/output/panels/` | DXF | `<stem>_panels_<timestamp>.dxf` |
| `_export_svg()` | `data/output/panels/` | SVG | `<stem>_panels_<timestamp>.svg` |

DXF and SVG exports call `unfold_mesh(mesh_current)` from `src.io.panel_exporter` before writing. All methods update `_export_status` with the result (success path or error message) for display in the UI.

#### `run()`

Registers `ui_callback` with Polyscope and calls `ps.show()`. Blocks until the viewer window is closed.

---

## Entry point — `main()`

```bash
# Default mesh (cylinder_10x8.obj if present, else file picker)
python src/visualisation/interactive_optimisation.py

# Explicit mesh file
python src/visualisation/interactive_optimisation.py data/input/generated/saddle_8x8.obj

# Reference dataset
python src/visualisation/interactive_optimisation.py data/input/reference_datasets/spot_quadrangulated.obj
```

`main()` performs the following steps:

1. Resolves the mesh path from `sys.argv[1]`, or falls back to `data/input/generated/cylinder_10x8.obj`, or opens a Tkinter file picker if available, or prints usage and exits.
2. Validates that the resolved path exists.
3. Prints the startup banner and viewer controls to stdout.
4. **Warms up Numba JIT kernels** — calls `warmup_numba_kernels()` from `src.backends` if Numba is available. This eliminates a 2–5 second stall that would otherwise occur on the first "Optimise" click while the JIT compiler runs. The warmup produces clear console feedback.
5. Constructs `InteractiveMeshOptimiser(mesh_path)` and calls `app.run()`.

---

## Requirements

Polyscope requires a display server. The viewer **will not run** in a headless CI environment.

```bash
pip install polyscope
```

Optional clipboard support on Linux:

```bash
sudo apt install xclip   # or xsel
```

---

## Results display — planarity thresholds

The results panel and `_build_results_text` apply the following thresholds consistently throughout the codebase:

| Threshold | Meaning |
| --------- | ------- |
| `< 1e-3` | Buildable — within real-world fabrication tolerance |
| `< 1e-2` | Near-flat — minor twist remaining |
| `< 0.1` | Slightly uneven — noticeable but not critical |
| `>= 0.1` | Needs further optimisation |

These thresholds match those used in `scripts/analysis/analyse_results.py` and the test suite, ensuring consistent reporting across all layers of the project.

---

## Viewer controls

| Action | Input |
| ------ | ----- |
| Rotate | Left mouse drag |
| Pan | Right mouse drag |
| Zoom | Scroll wheel |
| Select vertex/face | Left click on mesh |
| Clear selection | Handled automatically each frame |
