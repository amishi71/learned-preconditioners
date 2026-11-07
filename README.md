# Learned sparse preconditioners for conjugate gradient

A graph neural network (GNN) predicts a sparse lower-triangular factor **G** of an approximate inverse, `M⁻¹ = GᵀG`,
for symmetric positive-definite systems `Ax = b`. The preconditioner is SPD by construction (positive diagonal of G),
applies with two sparse mat-vecs, and the network is trained with a loss that needs no linear solve.

## Summary
- **What works.** Trained with the Kaporin loss, the GNN reproduces classical FSAI (the closed-form optimum on a fixed pattern)
  iteration-for-iteration on every matrix family it has seen, including a grid size it never trained on (Poisson 128²),
  and cuts CG iterations by
  roughly 40–47% versus no preconditioner on the Poisson family.
- **What does not.** It does **not beat** classical FSAI, IC(0) or AMG anywhere. On SuiteSparse Pres_Poisson every GNN variant fails
  (no convergence within 10·n iterations), and on the held-out Q1 hex FEM matrix the standard variants are *worse than no
  preconditioner*. A degree-robust variant (mean aggregation, log-scaled inputs) fixes the Q1 hex case (it matches FSAI) but not
  Pres_Poisson, where the inputs lie far outside the training range and the model is worse than Jacobi under its own training objective.
- **The honest framing.** Within a seen matrix family, a trained network can stand in for the per-row solves of FSAI at the cost
  of one forward pass; outside it, nothing in this work shows reliable generalization. See *Findings* and *Limitations*.

## Method
- **Scaling.** `Â = D^-1/2 A D^-1/2` (unit diagonal). The network outputs `G = I + ΔG`, so an untrained model *is* Jacobi.
- **Graph.** Homogeneous graph: node = row, edge = nonzero (row, column and value are always taken together from COO).
  Predicted entries live on the lower-triangle pattern of A (`tril`) or of A² (`tril2`, 2-hop).
- **Model.** Linear encoder → 3 residual GINE message-passing layers (edge feature = Â_ij) → edge MLP that predicts G_ij;
  the diagonal goes through a softplus. An optional `--robust` variant uses mean aggregation and log-scaled degree inputs.
- **Preconditioner.** `G_A = Ĝ·D^-1/2`, `M⁻¹ = G_Aᵀ G_A` (checked against FSAI computed directly on the unscaled matrix).
- **Losses.**
  - *Frobenius:* `‖GÂGᵀ − I‖²_F / n` (exact for n ≤ 1024, Hutchinson probes above).
  - *Kaporin:* `log(tr(GÂGᵀ)/n) − (2/n)Σ log g_ii`, which equals `log K + (1/n) log det Â` for the Kaporin conditioning number K.
    Classical FSAI is exactly the minimizer of K on a fixed pattern, so on `tril` the best a trained model can do is match FSAI.
    K is scale-invariant, so the network's overall scale is arbitrary (a diagonal near 0.6 instead of 1 is not a bug).
- **Baselines.** none, Jacobi, IC(0) (own implementation), classical FSAI on `tril` and `tril²`, and pyamg smoothed-aggregation AMG
  when available. `scipy.spilu` is deliberately **not** used with CG: its operator is non-symmetric (verified).

## Experimental setup
- **Training matrices (generated, 2D):** 5-point diffusion on 8²–64² grids with random anisotropy, variable coefficients with
  contrast up to 1000, Dirichlet or mixed boundaries. The *wide* distribution adds anisotropic 3D 7-point stencils and random
  Delaunay FEM meshes.
- **Test families:** training-like (Poisson 16²–128², anisotropic, random 2D), different structure (3D Poisson, Delaunay FEM),
  and **held out** even from the wide-trained models: a rotated-anisotropic 9-point stencil (positive off-diagonals,
  not an M-matrix), a Q1 hex FEM matrix (21 nonzeros per row), and SuiteSparse `ACUSIM/Pres_Poisson` (14,822 rows, 715,804 nonzeros,
  condition number about 2·10⁶, ~48 nonzeros per row).
- **Metric:** CG iterations to a relative residual of 1e-8 with a fixed random right-hand side per matrix (seeded by matrix name).
  Setup and per-apply times are also recorded in `benchmarks/results/eval_all.csv`, but iteration count is the primary metric:
  a Python pipeline is not compared by wall-clock against compiled libraries. Note that `tril²` costs about 2.3× the nonzeros of `tril`.
- **Checkpoints:** final weights after the last step (an early "best validation" checkpoint turned out to be misleading: it is the
  first step to reach the minimum on a plateau, and the validation matrix is in-distribution).

## Results
<!-- RESULTS:START -->
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
<!-- RESULTS:END -->

Figures: `benchmarks/results/spectrum_poisson16.png` (classical preconditioners), `benchmarks/results/spectrum_gnn.png` (GNN vs FSAI,
generated by `finalize.sh`), and `notebooks/results.ipynb` (all plots).

## Findings
1. **The Kaporin-trained GNN reproduces FSAI on seen families.** Iteration counts equal FSAI's to within a couple of percent on
   Poisson 16²–128² (128² is an unseen size), the random 2D family, anisotropic problems and 3D Poisson, and needs about 53–58% of the
   unpreconditioned iteration count (Poisson 16²–128²).
2. **It does not beat FSAI.** Ties within one or two iterations are noise. This is expected: FSAI is the exact optimum of the objective on
   `tril`. IC(0) beats everything on iterations, especially under strong anisotropy (about 6× fewer than the GNN on the anisotropic 64² problem). pyamg smoothed-aggregation AMG is far stronger still: 5–8 iterations on
   the Poisson family and 57 on Pres_Poisson (FSAI: 287), so no argument for the learned preconditioner can rest on iteration count.
3. **Kaporin loss beats Frobenius loss, mostly.** With final weights the Kaporin model needs fewer iterations than the Frobenius model
   on 11 of the 12 matrices where both converge (typically by 1–6%), and is far more robust on the held-out Q1 hex matrix (44 vs 5551
   iterations). The exception is Poisson 64², where the Frobenius model is 3% *below* FSAI (110 vs 113); that is the only place any model
   beat FSAI, it is not repeated at 16², 32² or 128² (+3%, +3%, +2%), and FSAI is not the Frobenius optimum, so it is not evidence of
   learning beyond FSAI. A first comparison on early checkpoints pointed the other way; that was a checkpoint-step confound.
   All of this is from one training seed.
4. **Loss as a proxy for iterations.** On 32 matrices × 6 FSAI-family preconditioners, the Kaporin value ranks preconditioners almost
   perfectly *within* a matrix (Spearman 0.998, min 0.986) versus 0.917 for the Frobenius loss. *Across* matrices neither is a good absolute
   predictor (pooled R² 0.19 and 0.27), so a pooled R² > 0.7 target is not met. The Frobenius loss can mis-rank: a 75% FSAI / 25% Jacobi
   blend has lower Frobenius loss than FSAI but needs more iterations.
5. **Wider pattern.** The `tril²` model beats classical `tril` FSAI, but only because it uses about 2.3× the nonzeros; against FSAI on the same
   `tril²` pattern it is within roughly 1–13% on training-like matrices and about 2× behind on the Delaunay FEM mesh.
6. **Generalization is limited to what the model has seen.**
   - On Delaunay FEM the narrow model is about 31% behind FSAI; training on FEM meshes (wide) closes only part of it (about 25%).
   - On Pres_Poisson all five GNN variants fail to converge within the iteration cap (the early checkpoints converged, very slowly).
     On Q1 hex FEM the standard variants are worse than no preconditioner (44 narrow, 350 wide, 5551 Frobenius, against 35 unpreconditioned).
   - `scripts/diagnose_ood.py`: Pres_Poisson rows have 17–49 neighbours and off-diagonal sums up to 5.6, against 2–4 and ≤ 1.4 in training.
     Measured under the training objective, the GNN's Kaporin K is 2.9× FSAI's there while Jacobi's is only 1.5×: a genuine generalization
     failure of the model, not an evaluation artefact.
   - Widening the training distribution was mixed: it helped the rotated-anisotropic held-out matrix (about 13% behind FSAI → about 3%)
     but made the Q1 hex held-out matrix much worse (44 → 350).
7. **Degree-robust variant (partial success).** Mean neighbour aggregation and log-scaled degree / off-diagonal-sum inputs, trained on the same
   wide distribution as the non-robust wide model, so that comparison changes only the architecture. On the held-out Q1 hex matrix it
   needs 22 iterations, identical to FSAI, against 350 for the non-robust wide model. Everywhere else it is indistinguishable from
   the non-robust wide model (rotated aniso 158 vs 157, FEM 157 vs 154, Poisson identical). It does **not** fix Pres_Poisson. The robust
   change bundles two modifications (aggregation and input scaling), so which one matters is not isolated; Pres_Poisson differs from
   the training data in more than row length (off-diagonal magnitudes, conditioning about 2·10⁶), which this change does not address. Measured under the training objective, the robust model's Kaporin K on Pres_Poisson is 2.4× FSAI's (non-robust: 2.9×; Jacobi: 1.5×), so it is still worse than Jacobi there. Single seed.
8. **Residual parametrization (`G = I + ΔG`).** Validation iterations on Poisson 32² are indistinguishable at 4000 steps with and without it;
   it only speeds up the first few hundred steps.

## Limitations
- **Single training seed** per configuration; differences of a few percent are not established. Only large effects (for example 45 vs 348
  iterations) should be read as real.
- **Generated training data.** Everything the model sees in training is synthetic 2D/3D diffusion or Delaunay FEM. Pres_Poisson is a
  single real matrix; one matrix does not establish behaviour on SuiteSparse in general.
- **Iterations, not time.** No claim is made about wall-clock speed-ups. Setup cost (a GNN forward pass) and per-apply cost are logged but
  are compared only inside one Python framework.
- **Baselines.** AMG (pyamg) is far ahead of every other preconditioner on iteration count; its setup and per-iteration cost are not
  compared here. PETSc (`petsc4py`/hypre) was not used; pyamg AMG and the in-repo IC(0) stand in. IC(0) is skipped above 4096 rows
  (pure-Python implementation) and can break down on non-M-matrices.
- **Hutchinson estimate** of the Kaporin trace sits inside a log and is slightly biased; the exact trace is used for n ≤ 1024.
- **Not covered:** GMRES / non-symmetric systems, matrices beyond ~15k rows, GPU timing, learned sparsity patterns.
- **Data access note.** `sparse.tamu.edu` returned HTTP 403 for downloads during this work; the matrix was obtained from the older
  University of Florida mirror (`scripts/fetch_suitesparse.py` tries it first).

## Reproduce
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib scikit-learn pytest jupyter ipykernel pyamg torch torch_geometric
pip install -e .
python scripts/run_tests.py                       # 35 tests (torch tests need torch + PyG)
python scripts/day3_checks.py                     # losses agree with numpy; overfit one matrix
python benchmarks/day1_baselines.py               # baseline table
python benchmarks/correlation_study.py            # loss-vs-iterations study
python scripts/fetch_suitesparse.py ACUSIM Pres_Poisson          # optional (see data access note)
python scripts/train.py --loss kaporin   --pattern tril  --steps 4000
python scripts/train.py --loss frobenius --pattern tril  --steps 4000
python scripts/train.py --loss kaporin   --pattern tril2 --steps 4000
python scripts/train.py --loss kaporin   --pattern tril  --dist wide --steps 4000
python scripts/train.py --loss kaporin   --pattern tril  --dist wide --robust --steps 4000
python scripts/train.py --loss kaporin   --pattern tril  --no_residual --steps 4000
bash scripts/finalize.sh                          # evaluate all final checkpoints, fill this README's tables, figures, notebook
python scripts/diagnose_ood.py --ckpt checkpoints/gnn_kaporin_tril_final.pt --mtx data/Pres_Poisson.mtx
bash scripts/clean_clone_check.sh                 # tests from a clean clone, no big files, no hardcoded paths or keys
```
Training on a laptop CPU takes on the order of tens of minutes per run; no GPU is needed.

## Layout
`src/lpc/` matrices, solvers (CG, IC(0)), fsai, graph, model, loss, precond, evaluate · `tests/` · `benchmarks/` (+ `results/`) ·
`scripts/` (train, evaluate, diagnose, finalize) · `notebooks/results.ipynb` · `logs/` training logs · `models/` final checkpoints

## References (verify before citing)
Kolotilina & Yeremin (1993) and Kaporin (1994) on FSAI; Grote & Huckle (1997) on SPAI; Häusner et al. (2023) neural incomplete
factorization; Li et al. (2023) learned preconditioners for CG; Davis & Hu (2011) SuiteSparse.
