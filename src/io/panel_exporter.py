"""
src/io/panel_exporter.py

Flat panel export for fabrication from optimised planar quad meshes.

After PQ optimisation, every quad face is approximately planar: all four
vertices lie on a common flat plane within a small tolerance determined by
the planarity residual. Each such planar quad can be unfolded into a 2D
panel by projecting its vertices onto their own best-fit plane and
expressing the projected coordinates in a local right-handed axis-aligned
frame. The resulting flat panels are the cutting templates that a fabricator
uses to manufacture the physical structure from sheet material.

This module provides the complete pipeline from a 'QuadMesh' to
fabrication-ready output in two standard formats:

  SVG -- Scalable Vector Graphics: readable in Inkscape, Adobe
         Illustrator, and any web browser. Suitable for visual inspection
         and laser cutting machines that accept vector input.

  DXF -- Drawing Exchange Format (R2010 / AC1024): the standard
         interchange format for CNC and laser cutting machines and
         architectural CAD tools (AutoCAD, Rhino, LibreCAD).

Mathematical unfolding procedure for a single quad face
--------------------------------------------------------
Given four 3D vertex positions v_0, v_1, v_2, v_3:

  Step 1: Compute the centroid c = mean(v_i).
  Step 2: Centre the vertices: w_i = v_i - c.
  Step 3: Fit a plane via SVD of the (4, 3) matrix [w_0; w_1; w_2; w_3].
          The plane normal n̂ is the right singular vector corresponding
          to the smallest singular value (the direction of minimum variance).
  Step 4: Define a local 2D frame on the plane:
              û = normalise(w_1 - w_0) (first edge direction)
              v̂ = n̂ × û (right-handed normal)
  Step 5: Project all four vertices into the frame:
              p_i = ((w_i · û), (w_i · v̂))
  Step 6: The planarity residual is max|w_i · n̂| over all four vertices
          (the largest out-of-plane distance before projection).

For a well-optimised mesh, the planarity residual is typically well below
1e-3 model units, which corresponds to less than 1 mm of warp for a typical
architectural panel of 1 m side length.

Note on self-containment
------------------------
The DXF writer is implemented from scratch as a minimal hand-built ASCII
writer and does not require the 'ezdxf' library. This keeps the dependency
footprint of the project minimal and guarantees compatibility with any
DXF-aware CNC controller.

References
----------
Liu, Y., Pottmann, H., Wallner, J., Yang, Y.-L., and Wang, W. (2006).
  "Geometric modelling with conical meshes and developable surfaces."
  ACM Transactions on Graphics, 25(3), pp. 681-689.

Pottmann, H., Eigensatz, M., Vaxman, A., and Wallner, J. (2015).
  "Architectural geometry." Computers and Graphics, 47, pp. 145-164.

Autodesk (2024). DXF R2010 Reference (AC1024).
  https://help.autodesk.com/view/OARX/2024/ENU/

Author: Muhammad Nabil
Date: March 2026
"""
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.core.mesh import QuadMesh

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class FlatPanel:
    """
    A single quad face from the mesh unfolded into a 2D flat panel.

    Each 'FlatPanel' corresponds to exactly one face in the source mesh and
    stores the 2D corner coordinates of the unfolded panel alongside the
    quality metrics needed to assess fabrication fitness.

    Attributes
    ----------
    face_id : int
        Index of the originating face in the mesh. Used to trace each panel
        back to its source face for inspection and labelling.
    vertices_2d : numpy.ndarray
        2D corner coordinates of the unfolded panel, shape (4, 2), dtype
        float64. Coordinates are expressed in the local panel frame whose
        origin is at the face centroid, in model units.
    planarity_residual : float
        Maximum absolute distance of any vertex from the best-fit plane
        before projection (model units). The quality metric for fabrication
        fitness: values below 1e-3 are considered within tolerance for
        architectural panel cutting.
    area_3d : float
        Approximate area of the face in 3D, computed via two-triangle split
        (model units squared). Used for material quantity estimation.
    area_2d : float
        Area of the unfolded 2D panel computed by the shoelace formula
        (model units squared). Should be approximately equal to 'area_3d'
        for near-planar faces; a large discrepancy indicates significant
        non-planarity.
    centroid_3d : numpy.ndarray
        3D centroid of the face, shape (3, ), dtype float64. Provided for
        traceability back to the source mesh geometry.
    normal : numpy.ndarray
        Unit normal of the best-fit plane, shape (3, ), dtype float64.
        Indicates the panel's orientation in 3D space, which is needed for
        structural analysis and panel connection design.
    """
    face_id: int
    vertices_2d: np.ndarray  # (4, 2)
    planarity_residual: float
    area_3d: float
    area_2d: float
    centroid_3d: np.ndarray  # (3,)
    normal: np.ndarray  # (3,)


@dataclass
class UnfoldReport:
    """
    Aggregate statistics from a batch unfolding operation.

    Produced by 'unfold_mesh()' and printed by its 'print()' method.
    Provides the key quality metrics needed to decide whether the mesh
    is ready for fabrication.

    Attributes
    ----------
    n_panels : int
        Total number of panels exported (equal to the number of mesh faces).
    max_planarity_residual : float
        Worst-case planarity residual across all panels (model units). The
        primary fabrication fitness metric: values below 1e-3 indicate that
        all panels are within architectural cutting tolerance.
    mean_planarity_residual : float
        Mean planarity residual across all panels (model units). A low mean
        with a high maximum indicates that a small number of faces are
        problematic and may benefit from targeted local refinement.
    max_area_distortion : float
        Maximum relative area change between 3D and 2D across all panels,
        expressed as a dimensionless ratio: |area_2d - area_3d| / area_3d.
        Values above 0.01 (1%) suggest that the projection is distorting
        the panel shape significantly, and further optimisation is advisable.
    warnings : list of str
        Non-fatal warnings generated during unfolding, typically listing the
        specific faces whose planarity residual exceeds the fabrication
        tolerance. Faces are listed by index up to a maximum of five; if
        more faces exceed tolerance, a summary count is appended instead.
    """
    n_panels: int = 0
    max_planarity_residual: float = 0.0
    mean_planarity_residual: float = 0.0
    max_area_distortion: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def print(self) -> None:
        print("=" * 60)
        print("UNFOLDING REPORT")
        print("=" * 60)
        print(f"  Panels exported       : {self.n_panels}")
        print(f"  Max planarity residual: {self.max_planarity_residual:.4e}")
        print(f"  Mean planarity residual:{self.mean_planarity_residual:.4e}")
        print(f"  Max area distortion   : {self.max_area_distortion * 100:.3f}%")
        if self.warnings:
            for w in self.warnings:
                print(f"  ⚠️  {w}")
        else:
            print("  ✓ No warnings — all panels within fabrication tolerance.")
        print("=" * 60 + "\n")


# ============================================================================
# CORE: UNFOLD A SINGLE QUAD FACE
# ============================================================================


def unfold_face(vertices_3d: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Project a single quad face onto its best-fit plane and return 2D coordinates.

    Computes the best-fit plane of the four 3D vertex positions using SVD,
    constructs a local right-handed 2D coordinate frame on the plane, and
    projects all four vertices into that frame. The projection discards the
    out-of-plane component; the size of that component is returned as the
    planarity residual.

    Parameters
    ----------
    vertices_3d : numpy.ndarray
        Vertex positions of a single quad face, shape (4, 3), dtype float64.

    Returns
    -------
    coords_2d : numpy.ndarray
        2D coordinates of the four vertices in the local panel frame,
        shape (4, 2), dtype float64. The frame origin is at the face centroid.
    planarity_residual : float
        Maximum absolute out-of-plane distance before projection (model units).
        This is the key fabrication quality metric for this face.
    normal : numpy.ndarray
        Unit normal of the best-fit plane, shape (3, ), dtype float64.

    Raises
    ------
    ValueError
        If 'vertices_3d' does not have a shape (4, 3).

    Notes
    -----
    The local frame û direction is set to the direction of the first edge
    (v_1 - v_0). If this edge is degenerate (length below 1e-12), the world
    x-axis is used instead. If the computed v̂ axis is also degenerate
    (normal parallel to û), a perpendicular fallback direction is chosen.
    These edge cases preserve correctness for degenerate faces, but their
    planarity residuals will typically be flagged by 'unfold_mesh()'.
    """
    if vertices_3d.shape != (4, 3):
        raise ValueError(f"Expected (4, 3) vertex array, got {vertices_3d.shape}.")

    centroid = vertices_3d.mean(axis=0)  # (3,)
    centred = vertices_3d - centroid  # (4, 3)

    # Best-fit plane normal via SVD
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    normal = Vt[-1]  # (3, ), smallest variance axis

    # Out-of-plane distances (planarity residual)
    out_of_plane = centred @ normal  # (4,)
    planarity_residual = float(np.max(np.abs(out_of_plane)))

    # Local 2D frame
    u_hat = centred[1] - centred[0]
    u_hat_norm = np.linalg.norm(u_hat)
    if u_hat_norm < 1e-12:
        u_hat = np.array([1.0, 0.0, 0.0])
    else:
        u_hat = u_hat / u_hat_norm

    v_hat = np.cross(normal, u_hat)
    v_hat_norm = np.linalg.norm(v_hat)
    if v_hat_norm < 1e-12:
        # Degenerate face — normal is parallel to u_hat; pick any perpendicular
        perp = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(u_hat, perp)) > 0.9:
            perp = np.array([0.0, 0.0, 1.0])
        v_hat = np.cross(normal, perp)
        v_hat = v_hat / (np.linalg.norm(v_hat) + 1e-12)
    else:
        v_hat = v_hat / v_hat_norm

    coords_2d = np.column_stack([centred @ u_hat, centred @ v_hat])  # (4, 2)

    return coords_2d, planarity_residual, normal


def _quad_area_2d(pts: np.ndarray) -> float:
    """
    Compute the area of a 2D polygon using the shoelace formula.

    Parameters
    ----------
    pts : numpy.ndarray
        Polygon vertices, shape (n, 2), dtype float64.

    Returns
    -------
    float
        Area of the polygon (always non-negative).
    """
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1]
        area -= pts[j, 0] * pts[i, 1]
    return abs(area) * 0.5


def _quad_area_3d(verts: np.ndarray) -> float:
    """
    Approximate the area of a 3D quad face by splitting it into two triangles.

    The quad is split along the diagonal v_0 to v_2. The total area is the
    sum of the areas of the two resulting triangles, each computed as half
    the size of the cross-product of its two-edge vectors.

    Parameters
    ----------
    verts : numpy.ndarray
        Vertex positions, shape (4, 3), dtype float64.

    Returns
    -------
    float
        Approximate area of the quad in 3D (model units squared).
    """
    a = 0.5 * np.linalg.norm(np.cross(verts[1] - verts[0], verts[2] - verts[0]))
    b = 0.5 * np.linalg.norm(np.cross(verts[2] - verts[0], verts[3] - verts[0]))
    return float(a + b)


# ============================================================================
# BATCH UNFOLD
# ============================================================================
def unfold_mesh(
    mesh: QuadMesh,
    planarity_tolerance: float = 1e-3,
    verbose: bool = True,
) -> Tuple[List[FlatPanel], UnfoldReport]:
    """
    Unfold every quad face of an optimised mesh into a flat 2D panel.

    Iterates over all faces in the mesh, calls 'unfold_face' for each,
    collects the results into 'FlatPanel' objects, computes per-face area
    distortion, and aggregates quality statistics into an 'UnfoldReport'.
    Faces whose planarity residual exceeds 'planarity_tolerance' are
    flagged with warnings in the report to alert the user that further
    optimisation may be needed before cutting.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh to unfold. Should be post-optimisation. Non-planar faces
        are not rejected but are flagged in the report.
    planarity_tolerance : float, optional
        Planarity residual threshold for issuing a warning per face.
        The default is 1e-3 (model units), which is the standard fabrication
        tolerance for architectural panel cutting.
    verbose : bool, optional
        If True, print the 'UnfoldReport' to standard output after
        unfolding. Default is True.

    Returns
    -------
    panels : list of FlatPanel
        One 'FlatPanel' per mesh face, in the same order as 'mesh.faces'.
    report : UnfoldReport
        Aggregate quality statistics for the entire unfolding operation.
    """
    panels: List[FlatPanel] = []
    residuals: List[float] = []
    area_distortions: List[float] = []
    warnings: List[str] = []

    if mesh.n_faces == 0:
        report = UnfoldReport(n_panels=0)
        if verbose:
            report.print()
        return panels, report

    _WARN_LIMIT = 5  # max per-face warnings before we summarise
    n_excess_warnings = 0

    for fid, face in enumerate(mesh.faces):
        verts_3d = mesh.vertices[face]  # (4, 3)
        coords_2d, residual, normal = unfold_face(verts_3d)

        a3 = _quad_area_3d(verts_3d)
        a2 = _quad_area_2d(coords_2d)
        distortion = abs(a2 - a3) / (a3 + 1e-12)

        if residual > planarity_tolerance:
            if len(warnings) < _WARN_LIMIT:
                warnings.append(
                    f"Face {fid}: planarity residual {residual:.4e} exceeds "
                    f"tolerance {planarity_tolerance:.1e} "
                    f"— optimise further before cutting."
                )
            else:
                n_excess_warnings += 1

        panels.append(
            FlatPanel(
                face_id=fid,
                vertices_2d=coords_2d,
                planarity_residual=residual,
                area_3d=a3,
                area_2d=a2,
                centroid_3d=verts_3d.mean(axis=0),
                normal=normal,
            )
        )
        residuals.append(residual)
        area_distortions.append(distortion)

    if n_excess_warnings > 0:
        warnings.append(
            f"... and {n_excess_warnings} more face(s) exceed tolerance "
            f"{planarity_tolerance:.1e}."
        )

    report = UnfoldReport(
        n_panels=len(panels),
        max_planarity_residual=float(np.max(residuals)),
        mean_planarity_residual=float(np.mean(residuals)),
        max_area_distortion=float(np.max(area_distortions)),
        warnings=warnings,
    )

    if verbose:
        report.print()

    return panels, report


# ============================================================================
# SVG EXPORT
# ============================================================================
def export_svg(
    panels: List[FlatPanel],
    filepath: str,
    panel_gap: float = 0.05,
    stroke_width: float = 0.002,
    colour_by_residual: bool = True,
    canvas_width_mm: float = 297.0,
    canvas_height_mm: float = 210.0,
) -> None:
    """
    Export flat panels to an SVG file arranged in a grid layout.

    Panels are arranged left-to-right, top-to-bottom in a grid sized to
    the aspect ratio of the canvas. Each panel is rendered as a closed
    polygon. When 'colour_by_residual=True', panels are coloured on a
    green-to-red gradient proportional to their planarity residual, giving
    a visual fabrication-readiness map at a glance.

    The SVG Y-axis points downward, so panel Y-coordinates are flipped
    during export. All coordinates are scaled to fit within the specified
    canvas dimensions.

    The file is written atomically via a temporary file and 'os.replace()',
    preventing a corrupt partial SVG if an exception occurs mid-write.

    Parameters
    ----------
    panels : list of FlatPanel
        The panels to export, as returned by 'unfold_mesh()'.
    filepath : str
        Output file path. Parent directories are created automatically.
    panel_gap : float, optional
        Gap between panels as a fraction of the maximum panel dimension.
        Default is 0.05 (5%).
    stroke_width : float, optional
        Panel edge line width in SVG units relative to the viewBox.
        Default is 0.002.
    colour_by_residual : bool, optional
        If True, colour panels from green (low residual) to red (high
        residual). If False, all panels are drawn in a uniform blue.
        Default is True.
    canvas_width_mm : float, optional
        SVG canvas width in millimetres. Default is 297 mm (A4 landscape).
    canvas_height_mm : float, optional
        SVG canvas height in millimetres. Default is 210 mm (A4 landscape).
    """
    if not panels:
        print("⚠️  No panels to export.")
        return

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine per-panel bounding boxes for layout
    bboxes = []
    for p in panels:
        mn = p.vertices_2d.min(axis=0)
        mx = p.vertices_2d.max(axis=0)
        bboxes.append((mn, mx))

    # Maximum residual for colour normalisation
    max_residual = max(p.planarity_residual for p in panels) + 1e-12

    # Grid layout: arrange panels left-to-right, top-to-bottom
    # Compute a uniform cell size from the max panel extents
    max_w = max((mx[0] - mn[0]) for mn, mx in bboxes)
    max_h = max((mx[1] - mn[1]) for mn, mx in bboxes)
    cell_w = max_w * (1.0 + panel_gap)
    cell_h = max_h * (1.0 + panel_gap)

    n_cols = max(1, int(canvas_width_mm / (cell_w * 1000 / canvas_width_mm + 1)))
    # Recompute for a sensible grid given canvas aspect ratio
    n_cols = max(
        1, int(np.ceil(np.sqrt(len(panels) * canvas_width_mm / canvas_height_mm)))
    )
    n_rows = int(np.ceil(len(panels) / n_cols))

    # Scale to fit the SVG viewBox
    total_w = n_cols * cell_w
    total_h = n_rows * cell_h
    scale = min(canvas_width_mm / total_w, canvas_height_mm / total_h)

    vb_w = canvas_width_mm
    vb_h = canvas_height_mm

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_width_mm}mm" height="{canvas_height_mm}mm" '
        f'viewBox="0 0 {vb_w:.3f} {vb_h:.3f}">',
        "  <title>PQ Mesh Flat Panels — Fabrication Layout</title>",
        "  <desc>Generated by PQ Mesh Optimisation Tool (University of Leeds)</desc>",
        "  <style>",
        f"    .panel-label {{ font: {scale * cell_h * 0.08:.2f}px sans-serif; "
        f"fill: #333; text-anchor: middle; dominant-baseline: central; }}",
        "  </style>",
    ]

    for idx, panel in enumerate(panels):
        row = idx // n_cols
        col = idx % n_cols

        # Cell origin (top-left of this panel's cell in SVG coords)
        ox = col * cell_w * scale
        oy = row * cell_h * scale

        # Centre panel within cell
        mn, mx = bboxes[idx]
        pan_w = mx[0] - mn[0]
        pan_h = mx[1] - mn[1]
        cx = ox + (cell_w * scale - pan_w * scale) / 2 - mn[0] * scale
        cy = oy + (cell_h * scale - pan_h * scale) / 2 + mx[1] * scale  # Y-flip

        # Build polygon points (SVG Y-axis is flipped)
        pts_str = " ".join(
            f"{cx + v[0] * scale:.4f},{cy - v[1] * scale:.4f}"
            for v in panel.vertices_2d
        )

        # Colour by planarity residual: green (0) → red (max)
        if colour_by_residual:
            t = panel.planarity_residual / max_residual  # 0…1
            r = int(255 * min(1.0, 2.0 * t))
            g = int(255 * min(1.0, 2.0 * (1.0 - t)))
            fill_colour = f"#{r:02x}{g:02x}00"
            fill_opacity = "0.35"
        else:
            fill_colour = "#b0c8e8"
            fill_opacity = "0.5"

        sw = stroke_width * scale * 20

        lines.append(
            f'  <polygon points="{pts_str}" '
            f'fill="{fill_colour}" fill-opacity="{fill_opacity}" '
            f'stroke="#1a1a2e" stroke-width="{sw:.4f}" />'
        )

        # Panel ID label at centroid
        label_x = cx + panel.vertices_2d[:, 0].mean() * scale
        label_y = cy - panel.vertices_2d[:, 1].mean() * scale
        lines.append(
            f'  <text x="{label_x:.3f}" y="{label_y:.3f}" '
            f'class="panel-label">P{panel.face_id}</text>'
        )

    lines.append("</svg>")

    # Atomic write — prevents a corrupt partial file if an exception occurs
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=out_path.parent, prefix=out_path.stem, suffix=".svg.tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        os.replace(tmp_path, out_path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    print(f"✓ SVG exported: {out_path}  ({len(panels)} panels)")


# ============================================================================
# DXF EXPORT
# ============================================================================
def export_dxf(
    panels: List[FlatPanel],
    filepath: str,
    scale_factor: float = 1000.0,
    layer_name: str = "PANELS",
) -> None:
    """
    Export flat panels to a DXF R2010 file for CNC or laser cutting.

    Each panel is written as a closed LWPOLYLINE entity. A companion MTEXT
    entity at the panel centroid labels the panel with its face index and
    planarity residual, enabling the fabricator to identify and inspect
    individual panels on the cutting table.

    The DXF is written as minimal hand-built plain ASCII R2010 (AC1024)
    without any external library dependency. This guarantees compatibility
    with AutoCAD, Rhino, LibreCAD, and most CNC controllers. All entities
    are placed on the Z=0 plane.

    The file is written atomically via a temporary file and 'os.replace()',
    preventing a corrupt partial DXF if an exception occurs mid-write.

    Parameters
    ----------
    panels : list of FlatPanel
        The panels to export, as returned by 'unfold_mesh()'.
    filepath : str
        Output file path. Parent directories are created automatically.
    scale_factor : float, optional
        Multiplier applied to all coordinates to convert model units to
        millimetres. Default is 1000.0 (assumes model coordinates in metres
        after normalisation by the preprocessor).
    layer_name : str, optional
        DXF layer name for all panel outlines and labels. Characters outside
        the DXF-legal set (alphanumeric, underscore, hyphen) are replaced
        with underscores. Maximum 255 characters. The default is 'PANELS'.
    """
    if not panels:
        print("⚠️  No panels to export.")
        return

    # Sanitise layer_name: DXF allows only alphanumeric, underscore, hyphen.
    # Reject or strip anything else to prevent injecting DXF control chars.
    sanitized_layer = re.sub(r"[^A-Za-z0-9_\-]", "_", layer_name)[:255]
    if sanitized_layer != layer_name:
        print(f"  ⚠️  layer_name {layer_name!r} was sanitized to {sanitized_layer!r}")
    layer_name = sanitized_layer

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s = scale_factor

    def dxf_header() -> List[str]:
        return [
            "  0\nSECTION",
            "  2\nHEADER",
            "  9\n$ACADVER",
            "  1\nAC1024",
            "  9\n$INSUNITS",
            " 70\n4",  # 4 = millimetres
            "  0\nENDSEC",
        ]

    def dxf_tables(layer: str) -> List[str]:
        return [
            "  0\nSECTION",
            "  2\nTABLES",
            "  0\nTABLE",
            "  2\nLAYER",
            " 70\n1",
            "  0\nLAYER",
            "  2\n" + layer,
            " 70\n0",
            " 62\n7",  # colour 7 = white/black
            "  6\nCONTINUOUS",
            "  0\nENDTAB",
            "  0\nENDSEC",
        ]

    def dxf_lwpolyline(pts: np.ndarray, layer: str) -> List[str]:
        """Closed LWPOLYLINE for a polygon."""
        lines = [
            "  0\nLWPOLYLINE",
            "  8\n" + layer,
            " 90\n" + str(len(pts)),  # number of vertices
            " 70\n1",  # 1 = closed
        ]
        for pt in pts:
            lines += [
                " 10\n" + f"{pt[0] * s:.6f}",
                " 20\n" + f"{pt[1] * s:.6f}",
            ]
        return lines

    def dxf_mtext(
        x: float, y: float, text: str, height: float, layer: str
    ) -> List[str]:
        return [
            "  0\nMTEXT",
            "  8\n" + layer,
            " 10\n" + f"{x:.6f}",
            " 20\n" + f"{y:.6f}",
            " 30\n0.0",
            " 40\n" + f"{height:.4f}",
            "  1\n" + text,
        ]

    all_lines: List[str] = []
    all_lines += dxf_header()
    all_lines += dxf_tables(layer_name)

    all_lines += [
        "  0\nSECTION",
        "  2\nENTITIES",
    ]

    label_height = 3.0  # mm

    for panel in panels:
        all_lines += dxf_lwpolyline(panel.vertices_2d, layer_name)

        cx = float(panel.vertices_2d[:, 0].mean()) * s
        cy = float(panel.vertices_2d[:, 1].mean()) * s
        label = f"P{panel.face_id} res={panel.planarity_residual:.2e}"
        all_lines += dxf_mtext(cx, cy, label, label_height, layer_name)

    all_lines += ["  0\nENDSEC", "  0\nEOF"]

    # Atomic write — prevents corrupt partial DXF if an exception occurs
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=out_path.parent, prefix=out_path.stem, suffix=".dxf.tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(all_lines) + "\n")
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    print(f"✓ DXF exported: {out_path}  ({len(panels)} panels)")


# ============================================================================
# CONVENIENCE: EXPORT BOTH FORMATS AT ONCE
# ============================================================================
def export_panels(
    mesh: QuadMesh,
    output_dir: str,
    stem: str = "mesh",
    planarity_tolerance: float = 1e-3,
    svg_colour_by_residual: bool = True,
    dxf_scale_mm: float = 1000.0,
    verbose: bool = True,
) -> UnfoldReport:
    """
    Convenience wrapper that unfolds a mesh and exports both SVG and DXF.

    Calls 'unfold_mesh()', 'export_svg()', and 'export_dxf()' in sequence.
    Output files are named '<stem>.svg' and '<stem>.dxf' inside 'output_dir'.
    This is the recommended entry point for the fabrication export workflow.

    Parameters
    ----------
    mesh : QuadMesh
        Post-optimisation mesh to export.
    output_dir : str
        Directory in which to write the output files. Created automatically
        if it does not exist.
    stem : str, optional
        Filename stem shared by both output files. Default is 'mesh'.
    planarity_tolerance : float, optional
        Planarity residual threshold passed to 'unfold_mesh()' for warning
        generation. Default is 1e-3.
    svg_colour_by_residual : bool, optional
        Passed to 'export_svg()'. Default is True.
    dxf_scale_mm : float, optional
        Scale factor passed to 'export_dxf()'. Default is 1000.0.
    verbose : bool, optional
        If True, print the unfolding report. Default is True.

    Returns
    -------
    UnfoldReport
        Aggregate quality statistics from the unfolding operation.

    Examples
    --------
    Complete fabrication export workflow::

        from src.io.obj_handler import load_obj
        from src.io.panel_exporter import export_panels
        from src.optimisation.optimiser import MeshOptimiser, OptimisationConfig

        mesh = load_obj("data/input/architectural/saddle_8x8.obj")
        config = OptimisationConfig(weights={"planarity": 10.0, "fairness": 1.0,
                                             "closeness": 5.0})
        result = MeshOptimiser(config).optimise(mesh)
        report = export_panels(result.optimised_mesh,
                               output_dir="results/panels",
                               stem="saddle_8x8")
    """
    panels, report = unfold_mesh(
        mesh,
        planarity_tolerance=planarity_tolerance,
        verbose=verbose,
    )

    base = Path(output_dir)
    export_svg(
        panels, str(base / f"{stem}.svg"), colour_by_residual=svg_colour_by_residual
    )
    export_dxf(panels, str(base / f"{stem}.dxf"), scale_factor=dxf_scale_mm)

    return report
