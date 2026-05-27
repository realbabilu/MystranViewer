"""
MYSTRAN Viewer — Dependency Installer
Run once:  python install_deps.py
           python install_deps.py --with-op2    (also installs pyNastran for OP2 support)
           python install_deps.py --check        (check what is already installed)
"""
import subprocess, sys, importlib, argparse

# ── Package definitions ────────────────────────────────────────────────────────
CORE = [
    # pip name           import name       why
    ("numpy",            "numpy",          "Array math foundation"),
    ("moderngl",         "moderngl",       "OpenGL 3.3 renderer"),
    ("glfw",             "glfw",           "Window / input handling"),
    ("pyrr",             "pyrr",           "3-D math (matrices, quaternions)"),
    ("imgui-bundle",     "imgui_bundle",   "Dear ImGui UI panels"),
]

OPTIONAL = [
    # pip name   import name    why
    ("pyNastran", "pyNastran",  "OP2/F06/BDF reading (needed for .op2 files)"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _installed(import_name: str) -> str | None:
    """Return version string if importable, else None."""
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "?")
    except ImportError:
        return None


def _install(pip_name: str) -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", pip_name],
        capture_output=True, text=True,
    )
    return r.returncode == 0, r.stdout + r.stderr


def _check_python():
    vi = sys.version_info
    if vi < (3, 10):
        print(f"  WARNING: Python {vi.major}.{vi.minor} detected.")
        print("  MYSTRAN Viewer requires Python 3.10+. Please upgrade.")
        sys.exit(1)
    print(f"  Python {vi.major}.{vi.minor}.{vi.micro}  ✓")


def _print_status(label, rows):
    pad = max(len(r[0]) for r in rows) + 2
    print(f"\n{'Package':<{pad}} {'Status':<14} Notes")
    print("-" * (pad + 30))
    for pip_name, import_name, note in rows:
        ver = _installed(import_name)
        status = f"installed v{ver}" if ver else "not installed"
        tick   = "✓" if ver else "✗"
        print(f"  {pip_name:<{pad-2}} {tick} {status:<12}  {note}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Install MYSTRAN Viewer dependencies")
    ap.add_argument("--with-op2",  action="store_true",
                    help="Also install pyNastran (required for .op2 file support)")
    ap.add_argument("--check",     action="store_true",
                    help="Only check what is installed, do not install anything")
    args = ap.parse_args()

    print("=" * 58)
    print("  MYSTRAN Viewer — Dependency Manager")
    print("=" * 58)

    print("\n[Python version]")
    _check_python()

    to_install = list(CORE)
    if args.with_op2:
        to_install += OPTIONAL

    if args.check:
        _print_status("Core packages",     CORE)
        _print_status("Optional packages", OPTIONAL)
        print()
        print("Tip: run without --check to install missing packages.")
        print("     run with --with-op2 to also install pyNastran.")
        return

    # ── Install ────────────────────────────────────────────────────────────────
    print("\n[Upgrading pip]")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                   capture_output=True)
    print("  done")

    failed = []
    print(f"\n[Installing {len(to_install)} package(s)]")
    for pip_name, import_name, note in to_install:
        already = _installed(import_name)
        if already:
            print(f"  {pip_name:<20} already v{already} — skipped")
            continue
        print(f"  {pip_name:<20} installing... ", end="", flush=True)
        ok, log = _install(pip_name)
        if ok:
            ver = _installed(import_name) or "?"
            print(f"OK  (v{ver})")
        else:
            print("FAILED")
            failed.append((pip_name, log[-400:]))

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    if failed:
        print("Some packages failed to install:")
        for name, log in failed:
            print(f"\n  {name}:")
            for line in log.strip().splitlines()[-6:]:
                print(f"    {line}")
        print()
        print("Troubleshooting tips:")
        print("  • moderngl/glfw need a GPU driver with OpenGL 3.3+ support.")
        print("  • On headless servers, OpenGL packages cannot be installed.")
        print("  • Try: pip install <package> --no-cache-dir")
        sys.exit(1)
    else:
        if not args.with_op2 and not _installed("pyNastran"):
            print("Note: pyNastran not installed.")
            print("      .op2 file loading will be unavailable.")
            print("      Re-run with:  python install_deps.py --with-op2")
        print()
        print("All packages installed successfully.")
        print()
        print("  To start the viewer:   python main.py")
        print("  To open a model:       python main.py path/to/model.bdf")


if __name__ == "__main__":
    main()
