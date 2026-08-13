# LookForge.spec — PyInstaller build for the desktop app.
# Build:  pyinstaller --noconfirm packaging/LookForge.spec
# Output: dist/LookForge.exe  (Windows) / dist/LookForge.app-style binary (macOS)
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Resolve paths relative to this spec file (PyInstaller sets SPECPATH).
_ROOT = os.path.dirname(SPECPATH)            # project root (parent of packaging/)
_data_dir = os.path.join(_ROOT, "src", "lutgen", "engine", "data")
datas = [
    (os.path.join(_data_dir, "base_dwg_di_to_rec709_g24.cube"), "lutgen/engine/data"),
    (os.path.join(_data_dir, "base_inverse_rec709_to_dwg_di.cube"), "lutgen/engine/data"),
]
binaries = []
hiddenimports = []

# colour-science ships datasets + many submodules — collect everything it needs.
for pkg in ("colour",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [os.path.join(SPECPATH, "run_gui.py")],
    pathex=[os.path.join(_ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Anything not on the app's real dependency tree is excluded outright — the first
    # build ran against a global Python and swallowed torch (251 MB), cv2 (99 MB),
    # pyarrow, onnxruntime, av, nltk, pandas, matplotlib (colour.plotting pulls it).
    excludes=["tkinter", "pytest", "matplotlib", "pandas", "torch", "cv2", "pyarrow",
              "onnxruntime", "av", "nltk", "blis", "IPython", "jupyter",
              "colour.examples", "colour.plotting"],
    cipher=block_cipher,
    noarchive=False,
)
# Trim Qt fat the hook over-includes: the 20 MB software-OpenGL fallback (any machine
# running Resolve has real GL) and the translation catalogues.
a.binaries = [b for b in a.binaries if "opengl32sw" not in b[0].lower()]
a.datas = [d for d in a.datas if "\\translations\\" not in d[0].lower()
           and "/translations/" not in d[0].lower()]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ReinaLook",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI app: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
