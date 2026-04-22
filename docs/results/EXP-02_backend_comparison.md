# EXP-02: Backend Comparison (NumPy vs Numba vs CuPy)

## Overview
This experiment quantifies the performance benefit of hardware-accelerated computation tiers (Numba JIT-compiled CPU parallelism and CuPy GPU execution) relative to the pure NumPy baseline. Understanding the speedup at each tier justifies the three-tier fallback architecture of the tool and demonstrates that the GPU acceleration investment yields meaningful returns for meshes of architectural relevance. It also validates that all three backends produce numerically equivalent results.

## Hardware Configuration
| Field   | Machine A (PC)                        |
|---------|---------------------------------------|
| CPU     | AMD Ryzen 7 5800X (8-core, 3.8 GHz)   |
| GPU     | NVIDIA RTX 3070 (8 GB VRAM, CUDA 12)  |
| RAM     | 16 GB DDR4-3600                       |
| OS      | Windows 11 (22H2)                     |
| Python  | 3.10.14                               |

## Experimental Setup
Five mesh sizes were selected to span the interactive and batch-processing regimes: 10×10 (100 faces), 20×20 (400 faces), 30×30 (900 faces), 50×50 (2,500 faces), and 75×75 (5,625 faces). Each mesh was optimised three times per backend. The Numba backend was pre-warmed with one excluded call before timing. The NumPy backend was forced by setting the environment variable `PQ_BACKEND=numpy`; Numba by `PQ_BACKEND=numba`; CuPy by `PQ_BACKEND=cupy`. Speedup is calculated as `time_numpy / time_backend`.

## Results

| Mesh Size | Vertices | Time NumPy (s) | Time Numba (s) | Time CuPy (s) | Speedup Numba (×) | Speedup CuPy (×) |
|-----------|----------|----------------|----------------|---------------|-------------------|------------------|
| 10×10     | 121      | 2.87           | 1.09           | 1.31          | 2.63×             | 2.19×            |
| 20×20     | 441      | 14.82          | 5.31           | 4.72          | 2.79×             | 3.14×            |
| 30×30     | 961      | 32.14          | 12.90          | 11.37         | 2.49×             | 2.83×            |
| 50×50     | 2,601    | 79.41          | 31.56          | 27.17         | 2.52×             | 2.92×            |
| 75×75     | 5,776    | 213.05         | 88.74          | 79.63         | 2.40×             | 2.68×            |

**Numerical agreement:** Maximum absolute difference in final vertex positions between NumPy and Numba: **3.2×10⁻⁷** (all sizes). Maximum absolute difference between NumPy and CuPy: **4.1×10⁻⁷** (all sizes). Both are well within the 1×10⁻⁶ tolerance specified in EXP-06.

## Plots
- **Figure 4.4:** Grouped bar chart — optimisation time per backend vs. mesh size — see `data/output/figures/EXP-02_backend_speedup_bar.png`

> **Figure numbering note:** An earlier draft of this file labelled the backend speedup bar chart
> as Figure 4.3. With the addition of the pipeline overview figure (Figure 4.1) in Chapter 4,
> and with EXP-01 occupying Figures 4.2 and 4.3, the backend comparison chart is correctly
> numbered **Figure 4.4** in the final dissertation. This file has been updated accordingly.

## Interpretation
Numba provides a consistent 2.4–2.8× speedup over NumPy across all mesh sizes, attributable to LLVM-compiled parallel loops (`@njit(parallel=True)` with `prange`) eliminating Python interpreter overhead and enabling multi-core utilisation. CuPy provides a 2.2–3.1× speedup, modestly outperforming Numba for larger meshes (≥20×20) where the GPU's parallelism is better utilised. Notably, for the smallest mesh tested (10×10, 121 vertices), CuPy is slightly *slower* than Numba: GPU data-transfer overhead (CPU→GPU memory copy) dominates for small payloads, a well-known characteristic of GPU computing. This empirically validates the three-tier fallback design: for small meshes the Numba tier is optimal, whilst for large meshes the CuPy tier provides the best throughput.

## Discussion
The absolute speedups (2–3×) are lower than might be expected for GPU acceleration. This is because the energy evaluation kernels involve relatively small, irregular memory access patterns (one kernel call per face), which do not fully saturate GPU memory bandwidth. A future architectural improvement would batch all face computations into a single CUDA kernel, which has been shown to yield 10–20× GPU speedups for similar mesh energy computations (Poranne et al., 2017). The current implementation prioritises correctness and maintainability over maximum GPU throughput, which is appropriate for an undergraduate dissertation deliverable. The numerical agreement results (max diff < 10⁻⁶) confirm that the three backends are interchangeable for all practical purposes, validating the floating-point stability of the energy formulations across hardware tiers.

## References Used
- Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A LLVM-based Python JIT compiler. *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC*, Article 7.
- Okuta, R. et al. (2017). CuPy: A NumPy-compatible library for NVIDIA GPU calculations. *Proceedings of the Workshop on ML Systems in NIPS 2017*.
- Poranne, R., Tarini, M., Huber, S., Panozzo, D., & Sorkine-Hornung, O. (2017). Autocuts: Simultaneous distortion and cut optimization for UV mapping. *ACM SIGGRAPH Asia*, 36(6), Article 215.
