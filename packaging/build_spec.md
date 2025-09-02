Packaging notes
--------------
- macOS (.app): pyinstaller or briefcase/pyoxidizer. For Apple Silicon, build on an M1/M2 or use cross-compilation.
- Windows (.exe): pyinstaller --onefile frontend/gui.py (include model separately)
- Keep model files out of the binary if large; provide a model manager to download or point to local path.
