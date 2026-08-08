# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Monad artifact: one file, no console banner, no sources.

Build:  pyinstaller monad_binary.spec
Run:    dist/monad_binary            # emits manifest hash, report hash, own digest

The manifest is regenerated here so the pinned digests always describe the code
being bundled; a stale manifest would make the startup check fail closed.
"""

import sys
from pathlib import Path

SPEC_DIR = Path(SPECPATH).resolve()

sys.path.insert(0, str(SPEC_DIR))
from monad_self_check import MANIFEST_NAME, write_manifest  # noqa: E402

MANIFEST = write_manifest(SPEC_DIR / MANIFEST_NAME)

a = Analysis(
    [str(SPEC_DIR / "monad_self_check.py")],
    pathex=[str(SPEC_DIR)],
    binaries=[],
    datas=[(str(MANIFEST), ".")],
    hiddenimports=["axiom_zero_engine"],
    hookspath=[],
    runtime_hooks=[],
    # The proof needs hashlib, json and fractions. Excluding the scientific
    # stack pulled in transitively by aleph_omega_kernel takes the artifact
    # from ~145 MB to ~10 MB; the kernel import then falls back to the literal
    # OMNIUM seed, which is bit-identical (339834188), so no digest moves.
    excludes=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "IPython",
        "sklearn",
        "pytest",
        "black",
        "cryptography",
        "zmq",
        "nbconvert",
        "nbformat",
        "jsonschema",
        "PIL",
        "gi",
        "tkinter",
        "setuptools",
        "pkg_resources",
    ],
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="monad_binary",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=True,
)
