#!/usr/bin/env bash
# Clone the committed repo into a temp dir and verify: tests pass, no large files, no hardcoded paths, no keys.
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"; TMP="$(mktemp -d)"; FAIL=0
git clone -q "$SRC" "$TMP/repo" || { echo "clone failed (commit first)"; exit 1; }
cd "$TMP/repo"
echo "== tests from clean clone"; python3 scripts/run_tests.py | tail -1 || FAIL=1
echo "== files > 5 MB"; BIG=$(find . -path ./.git -prune -o -type f -size +5M -print); [ -z "$BIG" ] && echo none || { echo "$BIG"; FAIL=1; }
echo "== hardcoded absolute paths"
P1='/home/'; P2='/Users/'; P3='C:\\\\'
H=$(grep -rnE "($P1|$P2|$P3)" --include='*.py' --include='*.sh' --include='*.md' --include='*.yml' --include='*.ipynb' . 2>/dev/null | grep -v clean_clone_check.sh | grep -v '^./.git/' || true)
[ -z "$H" ] && echo none || { echo "$H"; FAIL=1; }
echo "== possible keys/secrets"
K1='api[_-]?key'; K2='sk-[A-Za-z0-9]{20,}'; K3='AKIA[0-9A-Z]{12,}'; K4='secret'
S=$(grep -rniE "($K1|$K2|$K3|$K4)" --include='*.py' --include='*.sh' --include='*.yml' --include='*.ipynb' . 2>/dev/null | grep -v clean_clone_check.sh | grep -v '^./.git/' || true)
[ -z "$S" ] && echo none || { echo "$S"; FAIL=1; }
echo "== untracked data/checkpoints ignored"; git check-ignore -q data/x.mtx && git check-ignore -q checkpoints/x.pt && echo ok || FAIL=1
[ $FAIL -eq 0 ] && echo "CLEAN CLONE CHECK: PASS" || echo "CLEAN CLONE CHECK: FAIL"
exit $FAIL
