# Appendix B: External Materials

## B.1 Open-Source Software Libraries

All libraries listed below were used in their publicly released
versions as documented in the project `requirements.txt` file
in the repository root. All are released under permissive licences
permitting unrestricted academic use and redistribution.

| Library | Version | Licence | Purpose |
|:---|:---|:---|:---|
| NumPy | 1.26.4 | BSD-3-Clause | Array operations, energy/gradient computation |
| SciPy | 1.13.0 | BSD-3-Clause | L-BFGS-B optimiser (`scipy.optimize.minimize`) |
| Polyscope | 2.2.1 | MIT | 3D interactive visualisation and GUI |
| pytest | 8.1.1 | MIT | Test suite runner (229 tests, 20 modules) |
| Matplotlib | 3.8.4 | PSF/BSD | Convergence plots and figure generation |
| Numba | 0.59.1 | BSD-2-Clause | CPU-parallel JIT backend (Tier 2) |
| CuPy | 13.1.0 | MIT | CUDA GPU backend (Tier 1) |
| Trimesh | 4.3.2 | MIT | OBJ mesh loading and preprocessing |
| tqdm | 4.66.2 | MIT | Progress bar display during batch experiments |
| coverage | 7.4.4 | Apache-2.0 | Test coverage measurement (≥79% of `src/`) |

All libraries are widely used in academic and commercial settings
with permissive licences allowing unrestricted use. No modifications
were made to any library source code.

---

## B.2 Development Tools

| Tool | Version | Purpose |
|:---|:---|:---|
| Python | 3.12.2 | Primary programming language |
| Git | 2.44.0 | Version control for code and dissertation files |
| GitHub | N/A | Remote repository hosting and CI |
| Visual Studio Code | 1.88.1 | Primary IDE |
| Windows 11 (Machine A) | 23H2 | Primary development and benchmarking platform |
| macOS Sequoia (Machine B) | 15.3 | Secondary platform for cross-platform testing |
| CUDA Toolkit | 12.4 | GPU backend compilation and runtime (Machine A) |
| Conda | 24.1.2 | Environment management |

---

## B.3 Datasets

All datasets used in the real-world benchmark experiment (EXP-05)
are publicly available under permissive licences. Full provenance,
licence details, mesh statistics, and preprocessing outcomes are
documented in Appendix F.

| Model | File | Source | Author | Licence | URL |
|:---|:---|:---|:---|:---|:---|
| Spot | spot_quadrangulated.obj | Keenan's 3D Model Repository | Keenan Crane | CC0 (public domain) | https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/ |
| Blub | blub_quadrangulated.obj | Keenan's 3D Model Repository | Keenan Crane | CC0 (public domain) | https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/ |
| Bob | bob_quadrangulated.obj | Keenan's 3D Model Repository | Keenan Crane | CC0 (public domain) | https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/ |
| Oloid | oloid64_quad.obj | Keenan's 3D Model Repository | Keenan Crane | CC0 (public domain) | https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/ |

The Oloid dataset from Keenan Crane's repository provides the
developable oloid surface at multiple resolutions
(oloid64_quad.obj, oloid256_quad.obj, oloid1024_quad.obj,
oloid4096_quad.obj) as both quad and triangle meshes, with
accompanying generation code. The lowest-resolution quad variant
(oloid64_quad.obj) was selected for EXP-05 as the primary test
case, as its 64-face count is well within the scalability range
confirmed by EXP-01 whilst providing a geometrically meaningful
near-developable surface. No proprietary, sensitive, or personally
identifiable data was used at any point in this project.

---

## B.4 Code Repository

The complete source code, experiment scripts, test suite,
dissertation markdown files, and reproducibility documentation
for this project are publicly available at:

**Repository URL:**
https://github.com/nabiljefferson98/pq-mesh-optimisation-tool

**Licence:** MIT Licence

**Repository structure:**

| Directory / File | Contents |
|:---|:---|
| `src/` | Core pipeline source code (optimiser, gradients, mesh, backends, preprocessor, exporter, GUI) |
| `tests/` | 229-test suite across 20 modules (pytest) |
| `data/` | Benchmark OBJ files including Spot, Blub, Bob, Oloid |
| `scripts/` | Experiment runner scripts for EXP-01 to EXP-05 |
| `docs/dissertation/` | All dissertation chapter and appendix markdown files |
| `requirements.txt` | Pinned dependency versions |
| `README.md` | Setup, installation, and usage instructions |

The repository commit history provides a full version-controlled
record of all development decisions and documentation changes.
The dissertation supervisor and assessor may access all materials
through the public URL above without any authentication requirement.

---

## B.5 Use of Generative AI Tools

In accordance with the University of Leeds academic integrity policy
and the School of Computing guidance on the use of AI-assisted tools,
the following AI tools were used during this project:

| Tool | Provider | Purpose                                                                      |
|:---|:---|:-----------------------------------------------------------------------------|
| GitHub Copilot | GitHub / OpenAI | In-editor code completion suggestions during development, Dissertation drafting assistance, cross-reference audit, markdown generation                   |

**Usage statement:** AI-assisted tools were used to support the
drafting, structuring, and consistency-checking of the dissertation
text, and for code completion suggestions during implementation.
All AI-generated content — whether code or prose — was critically
reviewed by the author, verified against primary sources, and
substantially revised or rewritten before inclusion in any
deliverable. No AI-generated text was included verbatim in the
final report. No AI tool was used to generate experimental data,
run the optimisation pipeline, or produce any quantitative results
reported in Chapter 4.

---

**References**

Crane, K., Pinkall, U. and Schröder, P. (2013). Robust fairing
via conformal curvature flow. *ACM Transactions on Graphics*,
32(4), pp. 1–10.

Dirnböck, H. and Stachel, H. (1997). The development of the
oloid. *Journal for Geometry and Graphics*, 1(2), pp. 105–118.
