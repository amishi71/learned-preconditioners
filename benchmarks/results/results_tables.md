**CG iterations to relative residual 1e-8** (fewer is better; `n/a` = method not run / not applicable; `✗` = did not converge within 10·n iterations). Right-hand sides are seeded per matrix, so rows are comparable.

| Matrix | none | jacobi | IC(0) | AMG (pyamg) | FSAI (tril) | FSAI (tril²) | GNN Frobenius tril | GNN Kaporin tril² | GNN Kaporin tril | GNN Kaporin tril (wide) | GNN Kaporin tril (wide, robust) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Training-like** | | | | | | | | | | | |
| Poisson 16² | 52 | 52 | 20 | 6 | 30 | 23 | 31 | 23 | 30 | 30 | 30 |
| Poisson 32² | 101 | 101 | 34 | 7 | 58 | 41 | 60 | 43 | 58 | 58 | 58 |
| Poisson 64² | 199 | 199 | 65 | 8 | 113 | 81 | 110 | 83 | 113 | 113 | 113 |
| Poisson 128² (unseen size) | 379 | 379 | n/a | 8 | 201 | 153 | 206 | 158 | 201 | 202 | 202 |
| Anisotropic 64² (ay=0.01) | 366 | 366 | 32 | 31 | 195 | 136 | 204 | 166 | 197 | 198 | 196 |
| Random 2D family #0 | 150 | 150 | 50 | 7 | 81 | 61 | 82 | 63 | 81 | 81 | 81 |
| Random 2D family #1 | 210 | 209 | 62 | 8 | 111 | 79 | 114 | 88 | 111 | 111 | 111 |
| Random 2D family #2 | 198 | 198 | 65 | 8 | 113 | 80 | 116 | 83 | 113 | 114 | 113 |
| **Different structure (in training only for the wide-trained models)** | | | | | | | | | | | |
| 3D Poisson 16³ | 64 | 64 | 23 | 6 | 36 | 27 | 38 | 32 | 37 | 36 | 36 |
| Delaunay FEM, 4000 nodes | 660 | 293 | 102 | 21 | 123 | 73 | 166 | 163 | 161 | 154 | 157 |
| **Held out (denser / not M-matrix)** | | | | | | | | | | | |
| Rotated aniso 9-pt 64² (held out) | 305 | 305 | 18 | 19 | 153 | n/a | 181 | n/a | 173 | 157 | 158 |
| Q1 hex FEM 16³ (held out) | 35 | 35 | 16 | 5 | 22 | n/a | 5551 | n/a | 44 | 350 | 22 |
| SuiteSparse Pres_Poisson (held out) | 2856 | 746 | n/a | 57 | 287 | n/a | ✗ no conv. (148220) | n/a | ✗ no conv. (148220) | ✗ no conv. (148220) | ✗ no conv. (148220) |

**GNN vs classical FSAI on the same pattern, and vs unpreconditioned CG** (`+x%` = x% more iterations than FSAI; `⚠` = worse than no preconditioner)

| Matrix | GNN Frobenius tril | GNN Kaporin tril² | GNN Kaporin tril | GNN Kaporin tril (wide) | GNN Kaporin tril (wide, robust) |
|---|---|---|---|---|---|
| Poisson 16² | +3% | +0% | +0% | +0% | +0% |
| Poisson 32² | +3% | +5% | +0% | +0% | +0% |
| Poisson 64² | -3% | +2% | +0% | +0% | +0% |
| Poisson 128² (unseen size) | +2% | +3% | +0% | +0% | +0% |
| Anisotropic 64² (ay=0.01) | +5% | +22% | +1% | +2% | +1% |
| Random 2D family #0 | +1% | +3% | +0% | +0% | +0% |
| Random 2D family #1 | +3% | +11% | +0% | +0% | +0% |
| Random 2D family #2 | +3% | +4% | +0% | +1% | +0% |
| 3D Poisson 16³ | +6% | +19% | +3% | +0% | +0% |
| Delaunay FEM, 4000 nodes | +35% | +123% | +31% | +25% | +28% |
| Rotated aniso 9-pt 64² (held out) | +18% | n/a | +13% | +3% | +3% |
| Q1 hex FEM 16³ (held out) | +25132% ⚠ | n/a | +100% ⚠ | +1491% ⚠ | +0% |
| SuiteSparse Pres_Poisson (held out) | ✗ no conv. ⚠ | n/a | ✗ no conv. ⚠ | ✗ no conv. ⚠ | ✗ no conv. ⚠ |

- **GNN Frobenius tril** is worse than no preconditioner on: Q1 hex FEM 16³ (held out); SuiteSparse Pres_Poisson (held out).
- **GNN Kaporin tril** is worse than no preconditioner on: Q1 hex FEM 16³ (held out); SuiteSparse Pres_Poisson (held out).
- **GNN Kaporin tril (wide)** is worse than no preconditioner on: Q1 hex FEM 16³ (held out); SuiteSparse Pres_Poisson (held out).
- **GNN Kaporin tril (wide, robust)** is worse than no preconditioner on: SuiteSparse Pres_Poisson (held out).
