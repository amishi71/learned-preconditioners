#!/usr/bin/env bash
# Tries several SuiteSparse URLs/formats for Pres_Poisson; keeps the first real file (not an HTML error page).
cd "$(dirname "$0")/.." || exit 1
T=/tmp/ps_try
try() {
  url="$1"
  res=$(curl -4 -sSL --max-time 120 -o "$T" -w "%{http_code} %{size_download}" "$url" 2>&1)
  kind=$(file -b "$T" 2>/dev/null | cut -c1-60)
  echo "  -> $res | $kind"
  code=${res%% *}; size=${res##* }
  if [ "$code" = "200" ] && [ "$size" -gt 100000 ] 2>/dev/null && ! echo "$kind" | grep -qi html; then return 0; fi
  return 1
}
echo "== control: a tiny matrix from the same server (tells us whether the whole server refuses you)"
try "https://sparse.tamu.edu/MM/HB/bcsstk01.tar.gz" >/dev/null 2>&1; echo "control done:"; try "https://sparse.tamu.edu/MM/HB/bcsstk01.tar.gz"
for url in \
  "https://sparse.tamu.edu/MM/ACUSIM/Pres_Poisson.tar.gz" \
  "https://www.cise.ufl.edu/research/sparse/MM/ACUSIM/Pres_Poisson.tar.gz" \
  "https://suitesparse-collection-website.herokuapp.com/mat/ACUSIM/Pres_Poisson.mat" \
  "https://sparse.tamu.edu/mat/ACUSIM/Pres_Poisson.mat" ; do
  echo "== $url"
  if try "$url"; then
    case "$url" in
      *.tar.gz) cp "$T" data/Pres_Poisson.tar.gz && tar -xzf data/Pres_Poisson.tar.gz -C data && mv data/Pres_Poisson/Pres_Poisson.mtx data/ ;;
      *.mat)    cp "$T" data/Pres_Poisson.mat ;;
    esac
    echo "SUCCESS via $url"; exit 0
  fi
done
echo "ALL FAILED"; exit 1
