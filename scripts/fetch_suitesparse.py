"""Download one SuiteSparse matrix (Matrix Market) and check it is usable.
    python scripts/fetch_suitesparse.py ACUSIM Pres_Poisson
Needs a browser-like User-Agent (the default Python one gets HTTP 403). If it still fails, download the .tar.gz
from the matrix page in a browser and put Pres_Poisson.mtx in data/ - this script then just validates it."""
import io, pathlib, sys, tarfile, urllib.request

import numpy as np, scipy.io, scipy.sparse as sp, scipy.sparse.linalg as spla

group, name = sys.argv[1], sys.argv[2]
dest = pathlib.Path(__file__).resolve().parents[1] / "data"; dest.mkdir(exist_ok=True)
mtx = dest / f"{name}.mtx"
URLS = [f"https://www.cise.ufl.edu/research/sparse/MM/{group}/{name}.tar.gz",   # old UF mirror: worked when sparse.tamu.edu refused (HTTP 403)
        f"https://sparse.tamu.edu/MM/{group}/{name}.tar.gz",
        f"https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"]
if not mtx.exists():
    for url in URLS:
        try:
            print("downloading", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                tf = tarfile.open(fileobj=io.BytesIO(r.read()))
            member = next(m for m in tf.getmembers() if m.name.endswith(f"{name}.mtx"))
            member.name = f"{name}.mtx"; tf.extract(member, dest)
            break
        except Exception as e:
            print("  failed:", e)
    if not mtx.exists():
        sys.exit("download failed: fetch the Matrix Market .tar.gz in a browser, extract it, and put the .mtx in data/")
A = sp.csr_matrix(scipy.io.mmread(mtx))
sym = abs(A - A.T).max() / abs(A).max()
print(f"n={A.shape[0]} nnz={A.nnz} rel.asymmetry={sym:.1e} min diag={A.diagonal().min():.3e}")
b = np.random.default_rng(0).normal(size=A.shape[0]); its = [0]
x, info = spla.cg(A, b, rtol=1e-8, maxiter=20000, callback=lambda _: its.__setitem__(0, its[0] + 1))
print(f"unpreconditioned CG: converged={info == 0} iters={its[0]}")
print("OK to use" if sym < 1e-10 and A.diagonal().min() > 0 else "NOT SPD-looking: pick another matrix")
