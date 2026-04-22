# scripts/diagnostics — Mathematical Correctness Verification

These scripts verify that the energy function and gradient implementations
are mathematically correct **before** any benchmark or sensitivity results
are trusted. They should be run on both synthetic and real-world reference
meshes to confirm that correctness holds across different geometry types
and mesh resolutions.

Run these scripts at the start of the evaluation pipeline (Step 5) and
whenever any term in `src/optimisation/energy_terms.py` or
`src/optimisation/gradients.py` is modified.

---

## Script Reference

| Script | Experiment | What it does |
|---|---|---|
| `gradient_verification.py` | EXP-01 to EXP-05 (prerequisite) | Compares the analytical SVD-based gradient against a numerical finite-difference gradient for every face in the mesh. Reports the maximum and mean relative errors across all faces. A relative error below 1×10⁻⁵ confirms the gradient is correctly implemented to publication standard. |
| `energy_analysis.py` | EXP-04 (prerequisite) | Decomposes the total energy into its planarity (E\_p), fairness (E\_f), and closeness (E\_c) components for a given mesh, displays the per-component magnitudes, and recommends a set of starting weights via `suggest_weights_for_mesh()` based on the relative component scales. |

---

## Usage

```bash
# ── Gradient verification on synthetic mesh (default)
python scripts/diagnostics/gradient_verification.py

# ── Gradient verification on all four reference dataset meshes
python scripts/diagnostics/gradient_verification.py \
    --mesh data/input/reference_datasets/oloid/oloid256_quad.obj
python scripts/diagnostics/gradient_verification.py \
    --mesh data/input/reference_datasets/spot/spot_quadrangulated.obj
python scripts/diagnostics/gradient_verification.py \
    --mesh data/input/reference_datasets/blub/blub_quadrangulated.obj
python scripts/diagnostics/gradient_verification.py \
    --mesh data/input/reference_datasets/bob/bob_quad.obj

# ── Energy component analysis and weight recommendation
python scripts/diagnostics/energy_analysis.py
python scripts/diagnostics/energy_analysis.py \
    --mesh data/input/reference_datasets/spot/spot_quadrangulated.obj
```

---

## Correctness Thresholds

| Relative Error | Interpretation |
|---|---|
| < 1×10⁻⁵ | Publication-grade: gradient is analytically correct |
| 1×10⁻⁵ to 1×10⁻³ | Engineering-grade: acceptable for practical use but should be investigated |
| > 1×10⁻³ | Implementation error: do not trust optimisation results until resolved |

All five dissertation experiments require the gradient error to be below
1×10⁻⁵ on every mesh used. If the error exceeds this threshold after
modifying any source file, halt the pipeline and debug before continuing.

---

## When to Run

- At the start of every full evaluation run, before benchmarking.
- After any modification to `src/optimisation/energy_terms.py`,
  `src/optimisation/gradients.py`, or `src/core/quad_mesh.py`.
- Whenever the optimiser converges suspiciously fast, fails to converge,
  or produces geometrically implausible output.

See `docs/results/EXP-03_convergence.md` for the expected convergence
behaviour, and `docs/results/EXP-04_weight_sensitivity.md` for
guidance on selecting starting weights via `energy_analysis.py`.
