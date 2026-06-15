# AVISTA Project File Association

The installer registers `.avista` as the `AVISTA.Project` ProgID under the
Windows software classes registry.

The required open command is:

```text
"C:\Program Files\AVISTA\AVISTA.exe" "%1"
```

`%1` is quoted so project paths containing spaces are passed to AVISTA as one
argument. `main.py` validates `.avista` and legacy `.xtab` paths and loads the
project before constructing the main window.

The ProgID uses `AVISTA.exe,0` as its default icon. Inno Setup declares
`ChangesAssociations=yes`, creates the association during installation, and
removes the AVISTA ProgID during uninstall.

Verification:

1. Install AVISTA using the generated `AVISTA-Setup-<version>.exe`.
2. Double-click a valid `.avista` file.
3. Confirm the selected project is loaded.
4. Inspect `HKLM\Software\Classes\AVISTA.Project` on an administrative install.
5. Uninstall and confirm the AVISTA ProgID is removed.
