# scripts/mesh_generation

Mesh-generation scripts for building test datasets.

| Script | What it does |
| ------ | ------------ |
| `generate_test_meshes.py` | Generates parametric quad meshes (saddle, sphere cap, torus, Scherk, cylinder) and saves them as `.obj` files to `data/input/generated/`. |

## Quick start

```bash
# Generate all test meshes
python scripts/mesh_generation/generate_test_meshes.py
```

Output meshes are placed in `data/input/generated/` and are used as inputs for
the demo scripts and benchmarks.
