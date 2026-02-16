# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules
from pathlib import Path

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# Collect OpenVINO
# OpenVINO binaries are crucial
try:
    ov_datas, ov_binaries, ov_hiddenimports = collect_all('openvino')
    datas += ov_datas
    binaries += ov_binaries
    hiddenimports += ov_hiddenimports
    print("Collected OpenVINO")
except Exception as e:
    print(f"Error collecting OpenVINO: {e}")

# Collect Optimum Intel
try:
    opt_datas, opt_binaries, opt_hiddenimports = collect_all('optimum.intel')
    datas += opt_datas
    binaries += opt_binaries
    hiddenimports += opt_hiddenimports
    print("Collected Optimum Intel")
except Exception as e:
    print(f"Error collecting Optimum Intel: {e}")

# Collect Transformers
try:
    tr_datas, tr_binaries, tr_hiddenimports = collect_all('transformers')
    datas += tr_datas
    binaries += tr_binaries
    hiddenimports += tr_hiddenimports
    print("Collected Transformers")
except Exception as e:
    print(f"Error collecting Transformers: {e}")

# Collect other dependencies
deps = ['librosa', 'pyjson5', 'torch', 'faster_whisper', 'soundfile', 'scipy', 'sklearn']
for dep in deps:
    try:
        d, b, h = collect_all(dep)
        datas += d
        binaries += b
        hiddenimports += h
        print(f"Collected {dep}")
    except Exception as e:
        print(f"Error collecting {dep}: {e}")

# Add project source files
datas += [
    ('src', 'src'),
    ('generation_config.json5', '.'),
]

# Hidden imports that might be missed
hiddenimports += [
    'openvino.runtime',
    'openvino.runtime.utils',
    'openvino.utils',
    'optimum.intel.openvino',
    'optimum.intel.openvino.modeling_whisper',
    'transformers.models.whisper',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree',
    'sklearn.tree._utils',
]

a = Analysis(
    ['vino_v1.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'PyQt5', 'PySide2', 'matplotlib', 'IPython', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ChickenRice_v2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='transwithai.ico' if os.path.exists('transwithai.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ChickenRice_v2',
)
