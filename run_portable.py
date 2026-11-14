"""Run frozen Day 7 scripts with the bundled Day 6 adapter."""

import importlib
import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def prepare_runtime():
    load_dotenv(ROOT / ".env", override=False)

    runtime = ROOT / "runtime"
    sys.path.insert(0, str(runtime))

    # Preload the bundled package before frozen modules insert
    # the original machine-specific Day 6 path.
    module = importlib.import_module("src.store")
    actual = Path(module.__file__).resolve()
    expected = (runtime / "src/store.py").resolve()

    if actual != expected:
        raise RuntimeError(f"Unexpected database adapter: {actual}")

    sys.path.insert(0, str(ROOT / "starter"))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python run_portable.py <starter-script.py> [arguments]"
        )

    name = sys.argv.pop(1)
    if Path(name).name != name or not name.endswith(".py"):
        raise SystemExit("Provide only a starter script filename.")

    target = ROOT / "starter" / name
    if not target.is_file():
        raise SystemExit(f"Script not found: {name}")

    prepare_runtime()
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()