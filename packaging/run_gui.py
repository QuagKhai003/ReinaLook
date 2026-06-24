"""run_gui — PyInstaller entry point for the LookForge desktop app.

@context  A plain script PyInstaller can bundle; just launches the GUI.
@affects  Packaged by packaging/LookForge.spec. See README "Build the app".
"""

from lutgen.app.run import main

if __name__ == "__main__":
    raise SystemExit(main())
