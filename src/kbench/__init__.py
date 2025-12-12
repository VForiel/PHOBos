import warnings
import sys
import os
import toml

# Issue warning
warnings.warn("The 'kbench' package is deprecated. Please use 'phobos' instead.", DeprecationWarning, stacklevel=2)
if not os.environ.get('KBENCH_SUPPRESS_WARNING'):
    print("⚠️ 'kbench' is deprecated. Please use 'phobos' instead.")

# Import everything from phobos
try:
    from phobos import *
    from phobos import _SPHINX_BUILD, SANDBOX_MODE
    import phobos.classes as classes
    import phobos.modules as modules
    import phobos.sandbox as sandbox
    
    # Try to import optional modules that might be in phobos namespace
    try:
        from phobos import serial
    except ImportError:
        pass
        
    try:
        from phobos import xaosim
        from phobos import shm
    except ImportError:
        pass

except ImportError:
    print("❌ Error: 'phobos' package not found. Please install it.")

# Version
try:
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")
    if os.path.exists(pyproject_path):
        __version__ = toml.load(pyproject_path)["project"]["version"]
    else:
        __version__ = "unknown"
except Exception:
    __version__ = "unknown"
