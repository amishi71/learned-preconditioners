"""Minimal test runner for environments without pytest: python scripts/run_tests.py"""
import importlib.util, pathlib, sys, traceback, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
ok = bad = 0
for f in sorted(pathlib.Path(__file__).resolve().parents[1].joinpath("tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        print("IMPORT FAIL", f.name); traceback.print_exc(); bad += 1; continue
    for name in dir(mod):
        if name.startswith("test_"):
            try:
                getattr(mod, name)(); ok += 1; print("PASS", f.name, name)
            except unittest.SkipTest as e:
                print("SKIP", f.name, name, e)
            except Exception:
                bad += 1; print("FAIL", f.name, name); traceback.print_exc()
print(f"{ok} passed, {bad} failed"); sys.exit(1 if bad else 0)
