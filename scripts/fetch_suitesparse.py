"""Download one SuiteSparse matrix and check it is usable (symmetric, positive diagonal, CG converges).
    python scripts/fetch_suitesparse.py ACUSIM Pres_Poisson
Verify group/name/size on https://sparse.tamu.edu first: I could not check sizes (no network when authoring).
thermal1 (Schmid) is roughly 82k rows by my recollection - above the 50k range in your guide."""
import io, pathlib, sys, tarfile, urllib.request

import numpy as np, scipy.io, scipy.sparse as sp, scipy.sparse.linalg as spla

group, name = sys.argv[1], sys.argv[2]
url = f"https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"
dest = pathlib.Path(__file__).resolve().parents[1] / "data"
print("downloading", url)
with urllib.request.urlopen(url) as r:
    tf = tarfile.open(fileobj=io.BytesIO(r.read()))
member = next(m for m in tf.getmembers() if m.name.endswith(f"{name}.mtx"))
member.name = f"{name}.mtx"; tf.extract(member, dest)
A = sp.csr_matrix(scipy.io.mmread(dest / f"{name}.mtx"))
sym = abs(A - A.T).max() / abs(A).max()
print(f"n={A.shape[0]} nnz={A.nnz} rel.asymmetry={sym:.1e} min diag={A.diagonal().min():.3e}")
b = np.random.default_rng(0).normal(size=A.shape[0]); its = [0]
x, info = spla.cg(A, b, rtol=1e-8, maxiter=20000, callback=lambda _: its.__setitem__(0, its[0] + 1))
print(f"unpreconditioned CG: converged={info == 0} iters={its[0]}")
print("OK to use" if sym < 1e-10 and A.diagonal().min() > 0 else "NOT SPD-looking: pick another matrix")
