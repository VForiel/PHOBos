import sys
import os
import json

try:
    import kbench
except ImportError:
    print("❌ Error: kbench module not found.")
    sys.exit(1)

CONFIG_FILE = os.path.expanduser("~/.kbch.json")
if os.path.isfile(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {}

#==============================================================================
# Main
#==============================================================================

def main():

    # Check for valid command -------------------------------------------------

    if len(sys.argv) < 2:
        print("❌ Error: No equipment or option provided.")
        print("ℹ️ Use 'kbch --help' for usage information.")
        sys.exit(1)

    # Global help -------------------------------------------------------------

    if sys.argv[1] in ['--help', '-h']:
        show_help()
        sys.exit(0)

    # Version -----------------------------------------------------------------

    if sys.argv[1] in ['--version', '-v']:
        show_version()
        sys.exit(0)

    # Config ------------------------------------------------------------------

    if sys.argv[1] in ['config']:
        control_config(sys.argv[2:])

    # Mask wheel --------------------------------------------------------------

    if sys.argv[1] in ['mask']:
        control_mask(sys.argv[2:])
        sys.exit(0)

    # Invalid equipment -------------------------------------------------------

    print(f"❌ Error: Invalid equipment '{sys.argv[1]}'.")
    sys.exit(1)

#==============================================================================
# Tools
#==============================================================================

def show_help():
    print("Usage: kbench [equipment] [options | command]")
    print("       kbench config [options | command]")
    print("       kbench [options]")

    print("\nAvailable equipment:")
    print("  - mask")
    print("  - wheel (soon)")
    print("  - zab_h (soon)")
    print("  - zab_v (soon)")

    print("\nOptions:")
    print("  --help, -h     Show this help message")
    print("  --version, -v  Show version information (soon)")

def show_version():
    print("😅 Not implemented yet")
    pass

def get_config_file_path():
    if 'config_file' in CONFIG:
        path = CONFIG['config_file']
        return path if os.path.isfile(path) and path.endswith('.json') else None
    return None


#==============================================================================
# Config
#==============================================================================

def control_config(args):

    def show_config_help():
        print("Usage: kbench config [path]")
        print("       kbench config [options]")
        print("\nOptions:")
        print("  --help, -h     Show this help message")
        print("  --reset, -r    Reset configuration to default")
        print("  --show, -s     Show current configuration")

    # Invalid command ---------------------------------------------------------

    if len(args) < 1:
        print("❌ Error: No config provided.")
        print("ℹ️ Use 'kbch config --help' for usage information.")
        sys.exit(1)

    # Help --------------------------------------------------------------------

    if args[0] in ['--help', '-h']:
        show_config_help()
        sys.exit(0)

    # Set ---------------------------------------------------------------------

    if os.path.exists(args[0]) and args[0].endswith('.json'):
        print(f"⌛ Updating configuration path...")
        CONFIG['config_path'] = os.path.abspath(args[0])
        with open(CONFIG_FILE, "w") as f:
            json.dump(CONFIG, f, indent=4)
        print("✅Done")
        sys.exit(0)

    # Reset -------------------------------------------------------------------

    if args[0] in ['--reset', '-r']:
        print(f"⌛ Resetting configuration to default...")
        CONFIG.pop('config_path')
        with open(CONFIG_FILE, "w") as f:
            json.dump(CONFIG, f, indent=4)
        print("✅ Done")
        sys.exit(0)

    # Show --------------------------------------------------------------------

    if args[0] in ['--show', '-s']:
        if 'config_path' in CONFIG:
            print(f"Current config file: {CONFIG['config_path']}")
        else:
            print("🫤 No configuration file set.")
        sys.exit(0)

    # Invalid args ------------------------------------------------------------

    print(f"❌ Error: Invalid config path. The path should point to a .json file.")
    print("ℹ️ Use 'kbch config --help' for usage information.")
    sys.exit(1)

#==============================================================================
# Control mask wheel
#==============================================================================

def control_mask(args):

    masks = {
        0: 'dot',
        1: 'bigdot',
        2: 'none',
        3: 'asgard',
        4: 'line',
        5: 'weird'
    }

    def show_help():
        print("Usage: kbench mask [mask]")
        print("       kbench mask [option]")
        show_available_masks()
        print("\nOptions:")
        print("  --list, -l   Show available masks")
        print("  --help, -h   Show this help message")

    def show_available_masks():
        print("\nAvailable masks:")
        for idx, name in masks.items():
            print(f"  {idx}: {name}")

    # No command --------------------------------------------------------------

    if len(args) < 1:
        print("❌ Error: No mask provided.")
        print("ℹ️ Use 'kbch mask --help' for usage information.")
        sys.exit(1)

    # Help --------------------------------------------------------------------

    if args[0] in ['--help', '-h']:
        show_help()
        sys.exit(0)

    # List --------------------------------------------------------------------

    if args[0] in ['--list', '-l']:
        show_available_masks()
        sys.exit(0)

    # Set ---------------------------------------------------------------------

    # If the mask is designated by it's name
    if args[0] in masks.values():
            mask = list(masks.keys())[list(masks.values()).index(args[0])]

    # Else if the mask is designated by it's index
    else:
        try:
            mask = int(args[0])
        except (IndexError, ValueError):
            print("❌ Error: Invalid mask value.")
            show_available_masks()
            sys.exit(1)

    print(f"⌛ Setting mask: {masks[mask]}...")
    
    p = kbench.PupilMask()

    p.apply_mask(mask, config=get_config_file_path())

    print("✅ Done")

#==============================================================================
# EOF
#==============================================================================

if __name__ == "__main__":
    main()
    sys.exit(0)