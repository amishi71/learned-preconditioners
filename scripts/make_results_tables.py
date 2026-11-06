"""Turn benchmarks/results/eval_all.csv into markdown tables and (optionally) inject them into the README.

    python scripts/make_results_tables.py --readme README.md

Replaces everything between <!-- RESULTS:START --> and <!-- RESULTS:END --> in the README, and always writes
benchmarks/results/results_tables.md. Nothing is typed by hand: re-run after every evaluation."""
import argparse, pathlib, re

import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"

LABELS = {
    "poisson16": "Poisson 16²", "poisson32": "Poisson 32²", "poisson64": "Poisson 64²", "poisson128": "Poisson 128² (unseen size)",
    "aniso64_0.01": "Anisotropic 64² (ay=0.01)", "aug0": "Random 2D family #0", "aug1": "Random 2D family #1",
    "aug2": "Random 2D family #2", "ood_poisson3d_16": "3D Poisson 16³", "ood_fem_delaunay_4000": "Delaunay FEM, 4000 nodes",
    "heldout_rotaniso9_64": "Rotated aniso 9-pt 64² (held out)", "heldout_q1hex_16": "Q1 hex FEM 16³ (held out)",
    "suitesparse": "SuiteSparse Pres_Poisson (held out)",
}
GROUPS = [
    ("Training-like", ["poisson16", "poisson32", "poisson64", "poisson128", "aniso64_0.01", "aug0", "aug1", "aug2"]),
    ("Different structure (in training only for the wide-trained models)", ["ood_poisson3d_16", "ood_fem_delaunay_4000"]),
    ("Held out (denser / not M-matrix)", ["heldout_rotaniso9_64", "heldout_q1hex_16", "suitesparse"]),
]
BASE_ORDER = ["none", "jacobi", "ic0", "pyamg_amg", "fsai_tril", "fsai_tril2"]


def pretty(m: str) -> str:
    if m == "pyamg_amg": return "AMG (pyamg)"
    if m == "ic0": return "IC(0)"
    if m == "fsai_tril": return "FSAI (tril)"
    if m == "fsai_tril2": return "FSAI (tril²)"
    if m.startswith("gnn_"):
        core = m[4:].replace("_final", "")
        loss, pat, *rest = core.split("_")
        tags = {"wide": "wide", "robust": "robust", "nores": "no-residual"}
        extra = ", ".join(tags.get(r, r) for r in rest)
        return f"GNN {loss.capitalize()} {pat.replace('tril2', 'tril²')}" + (f" ({extra})" if extra else "")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RES / "eval_all.csv"))
    ap.add_argument("--readme", default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    if "converged" not in df:
        df["converged"] = True
    methods = [m for m in BASE_ORDER if m in set(df.method)] + sorted(m for m in set(df.method) if m.startswith("gnn_"))
    gnns = [m for m in methods if m.startswith("gnn_")]
    it = df.pivot_table(index="matrix", columns="method", values="iters", aggfunc="first")
    cv = df.pivot_table(index="matrix", columns="method", values="converged", aggfunc="first")

    def cell(mat, m):
        v = it.at[mat, m] if (mat in it.index and m in it.columns) else np.nan
        if pd.isna(v): return "n/a"
        ok = bool(cv.at[mat, m]) if pd.notna(cv.at[mat, m]) else True
        return f"{int(v)}" if ok else f"✗ no conv. ({int(v)})"

    out = []
    out.append("**CG iterations to relative residual 1e-8** (fewer is better; `n/a` = method not run / not applicable; "
               "`✗` = did not converge within 10·n iterations). Right-hand sides are seeded per matrix, so rows are comparable.\n")
    out.append("| Matrix | " + " | ".join(pretty(m) for m in methods) + " |")
    out.append("|---|" + "---|" * len(methods))
    for gname, mats in GROUPS:
        out.append(f"| **{gname}** |" + " |" * len(methods))
        for mat in mats:
            if mat in it.index:
                out.append(f"| {LABELS.get(mat, mat)} | " + " | ".join(cell(mat, m) for m in methods) + " |")
    notes = []
    if gnns:
        out.append("\n**GNN vs classical FSAI on the same pattern, and vs unpreconditioned CG** "
                   "(`+x%` = x% more iterations than FSAI; `⚠` = worse than no preconditioner)\n")
        out.append("| Matrix | " + " | ".join(pretty(m) for m in gnns) + " |")
        out.append("|---|" + "---|" * len(gnns))
        worse = {m: [] for m in gnns}
        for gname, mats in GROUPS:
            for mat in mats:
                if mat not in it.index: continue
                row = []
                for m in gnns:
                    ref = "fsai_tril2" if "tril2" in m else "fsai_tril"
                    v, f, n0 = it.at[mat, m], it.at[mat, ref] if ref in it.columns else np.nan, it.at[mat, "none"] if "none" in it.columns else np.nan
                    ok = bool(cv.at[mat, m]) if pd.notna(cv.at[mat, m]) else True
                    if pd.isna(v) or pd.isna(f):
                        row.append("n/a"); continue
                    s = f"{(v / f - 1) * 100:+.0f}%"
                    if not ok: s = "✗ no conv."
                    if (not ok) or (pd.notna(n0) and v > n0):
                        s += " ⚠"; worse[m].append(LABELS.get(mat, mat))
                    row.append(s)
                out.append(f"| {LABELS.get(mat, mat)} | " + " | ".join(row) + " |")
        for m in gnns:
            if worse[m]:
                notes.append(f"- **{pretty(m)}** is worse than no preconditioner on: " + "; ".join(worse[m]) + ".")
        if notes:
            out.append("\n" + "\n".join(notes))
    text = "\n".join(out) + "\n"
    (RES / "results_tables.md").write_text(text)
    print(text)
    if args.readme:
        p = pathlib.Path(args.readme); s = p.read_text()
        pat = re.compile(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.S)
        assert pat.search(s), "README has no RESULTS markers"
        p.write_text(pat.sub(lambda m: m.group(1) + "\n" + text + m.group(2), s))
        print(f"README updated: {args.readme}")


if __name__ == "__main__":
    main()
