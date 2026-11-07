#!/usr/bin/env bash
# One command after training: evaluate every final checkpoint -> tables -> README -> figures -> notebook -> tests.
#   bash scripts/finalize.sh                 (run from the project root, venv active)
#   bash scripts/finalize.sh --report-only   (skip the slow evaluation; rebuild tables, figures, notebook from the existing CSV)
set -e
cd "$(dirname "$0")/.."
if [ "${1:-}" != "--report-only" ]; then
  FLAGS=""; for f in checkpoints/*_final.pt; do FLAGS="$FLAGS --ckpt $f"; done
  [ -z "$FLAGS" ] && { echo "no checkpoints/*_final.pt found - train first"; exit 1; }
  EXTRA=""
  [ -f data/Pres_Poisson.mtx ] && EXTRA="$EXTRA --mtx data/Pres_Poisson.mtx"
  python -c "import pyamg" 2>/dev/null && EXTRA="$EXTRA --amg"
  echo "== evaluating:$FLAGS $EXTRA"
  python scripts/evaluate_checkpoint.py $FLAGS $EXTRA 2>&1 | grep -v -i -E "warn|spmatrix|sparse array|sparray|default return|Unless you|on 2D shapes|migration"
else
  echo "== --report-only: reusing benchmarks/results/eval_all.csv"
fi
echo "== tables -> README"
python scripts/make_results_tables.py --readme README.md > /dev/null && echo done
echo "== spectrum figure"
SF=""; for f in checkpoints/gnn_kaporin_tril_final.pt checkpoints/gnn_kaporin_tril_wide_robust_final.pt; do [ -f "$f" ] && SF="$SF --ckpt $f"; done
[ -n "$SF" ] && python benchmarks/gnn_spectrum_figure.py $SF 2>&1 | grep -v -i warn
echo "== notebook"
jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb 2>&1 | tail -2
python - <<'PY'
# executed notebooks embed absolute paths in warnings/tracebacks: replace the home directory by ~
import pathlib
p = pathlib.Path("notebooks/results.ipynb"); h = str(pathlib.Path.home())
p.write_text(p.read_text().replace(h, "~"))
PY
echo "== tests"
python scripts/run_tests.py 2>&1 | grep -E "FAIL|passed"
