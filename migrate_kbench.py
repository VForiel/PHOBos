import os

PHOBOS_ROOT = "src/phobos"
KBENCH_ROOT = "src/kbench"

for root, dirs, files in os.walk(PHOBOS_ROOT):
    for file in files:
        if not file.endswith(".py"):
            continue
            
        rel_path = os.path.relpath(os.path.join(root, file), PHOBOS_ROOT)
        kbench_path = os.path.join(KBENCH_ROOT, rel_path)

        if file == "__init__.py":
            if root == PHOBOS_ROOT:
                continue # Skip root __init__.py
            # For subpackages, we also want to redirect
            # src/phobos/classes/__init__.py -> phobos.classes
            module_path = "phobos." + os.path.relpath(root, PHOBOS_ROOT).replace(os.sep, ".")
            
            print(f"Redirecting {kbench_path} to {module_path}")
            
            content = f"""from {module_path} import *
import warnings
# warnings.warn("This package is deprecated. Use {module_path} instead.", DeprecationWarning, stacklevel=2)
"""
            with open(kbench_path, "w") as f:
                f.write(content)
            continue
            
        if os.path.exists(kbench_path):
            # Construct module path
            # src/phobos/classes/deformable_mirror.py -> phobos.classes.deformable_mirror
            module_path = "phobos." + rel_path.replace(os.sep, ".")[:-3]
            
            print(f"Redirecting {kbench_path} to {module_path}")
            
            content = f"""from {module_path} import *
import warnings
warnings.warn("This module is deprecated. Use {module_path} instead.", DeprecationWarning, stacklevel=2)
"""
            with open(kbench_path, "w") as f:
                f.write(content)

