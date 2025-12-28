# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

def add_pkg(pkgname):
    # collect_all returns: (datas, binaries, hiddenimports)
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkgname)
    return pkg_datas, pkg_binaries, pkg_hidden

datas = []
binaries = []
hiddenimports = []

# Streamlit + Chroma + your parsers
for pkg in ["streamlit", "chromadb", "pypdf", "docx", "google", "rfc3987", "rfc3987_syntax"]:
    d, b, h = add_pkg(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Include app.py into the bundle so run_app.py can load it from _MEIPASS
datas += [("app.py", ".")]

# Optional: avoid warnings / optional deps you don't use
excludes = [
    "langchain", "streamlit.external.langchain",
    "fastapi", "chromadb.server.fastapi"
]

a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RAG-Vault",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # set True if you want console window
    icon="rag_vault.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="RAG-Vault",
)
