import warnings
import sys
from phobos.scripts.cli import main as phobos_main

def main():
    print("⚠️ 'kbch' command is deprecated. Please use 'phob' instead.")
    warnings.warn("'kbch' is deprecated. Use 'phob'.", DeprecationWarning, stacklevel=2)
    phobos_main()

if __name__ == "__main__":
    main()
