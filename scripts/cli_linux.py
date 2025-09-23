import sys
import os
import json
import yaml

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
        sys.exit(0)

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
    print("Usage: kbch [equipment] [options | command]")
    print("       kbch config [options | command]")
    print("       kbch [options]")

    print("\nAvailable equipment:")
    print("  - mask")
    print("  - filter")

    print("\nOptions:")
    print("  --help, -h     Show this help message")
    print("  --version, -v  Show version information (soon)")

def show_version():
    print("😅 Not implemented yet")
    pass

def get_config_file_path():
    if 'config_path' in CONFIG:
        path = CONFIG['config_path']
        return path if os.path.isfile(path) and (path.endswith('.yml') or path.endswith('.yaml')) else None
    return None

def is_config_set():
    return bool(get_config_file_path())

def get_config():
    if not is_config_set():
        print("❌ Error: No valid configuration file set.")
        print("ℹ️ Use 'kbch config --help' for usage information.")
        sys.exit(1)
    with open(get_config_file_path(), 'r') as f:
        config = yaml.safe_load(f)
    return config

#==============================================================================
# Config
#==============================================================================

def control_config(args):

    def show_config_help():
        print("Usage: kbch config [path]")
        print("       kbch config [options]")
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

    if is_config_set():
        config = get_config()

        masks = {}
        for key, value in config['mask']['slots'].items():
            masks[int(key)] = value['name']

    else:
        masks = {
            'dot':{
                'x': 0,
                'y': 0,
                'a': 0,
            },
            'bigdot':{
                'x': 0,
                'y': 0,
                'a': 60,
            },
            'none':{
                'x': 0,
                'y': 0,
                'a': 120,
            },
            'asgard':{
                'x': 0,
                'y': 0,
                'a': 180,
            },
            'line':{
                'x': 0,
                'y': 0,
                'a': 240,
            },
            'weird':{
                'x': 0,
                'y': 0,
                'a': 300,
            },
        }

    def show_help():
        print("Usage: kbch mask set [mask]")
        print("       kbch mask mvh [-a|--abs] [value (int)]")
        print("       kbch mask mvv [-a|--abs] [value (int)]")
        print("       kbch mask mva [-a|--abs] [value (float)]")
        print("       kbch mask [option]")
        show_available_masks()
        print("\nOptions:")
        print("  --list, -l   Show available masks")
        print("  --help, -h   Show this help message")

    def show_available_masks():
        print("\nAvailable masks:")
        for name in masks.keys():
            print(f"  {name}")

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

    if args[0] in ['set']:

        if len(args) < 2:
            print("❌ Error: No mask provided.")
            show_available_masks()
            sys.exit(1)

        if args[1] in masks:
            mask = args[1]
        else:
            print("❌ Error: Invalid mask value.")
            show_available_masks()
            sys.exit(1)

        print(f'⌛ Setting "{mask}" mask...')
        
        p = kbench.PupilMask(
            newport_port=config['mask']['ports']['newport'],
            zaber_port=config['mask']['ports']['zaber']
        )

        if config:
            p.apply_mask(mask, config=get_config_file_path())
        else:
            p.rotate(masks[mask]['a'], abs=True)
            p.move_h(masks[mask]['x'], abs=True)
            p.move_v(masks[mask]['y'], abs=True)

        print("✅ Done")
        sys.exit(0)

    # Move --------------------------------------------------------------------

    if args[0] in ['mvh', 'mvv', 'mva']:

        print(f"⌛ Moving mask...")

        p = kbench.PupilMask(
            newport_port=config['mask']['ports']['newport'],
            zaber_port=config['mask']['ports']['zaber']
        )

        try:
            if len(args) > 1 and args[1] in ['-a', '--abs']:
                abs = True
                value = float(args[2])
            else:
                abs = False
                value = float(args[1])
        except (IndexError, ValueError):
            print("❌ Error: Invalid move value.")
            sys.exit(1)

        if args[0] == 'mvh':
            p.move_h(int(value), abs=abs)
        elif args[0] == 'mvv':
            p.move_v(int(value), abs=abs)
        elif args[0] == 'mva':
            p.move_a(value, abs=abs)

        print("✅ Done")
        sys.exit(0)

#==============================================================================
# Control filter wheel
#==============================================================================

def control_filter(args):
    def show_help():
        print("Usage: kbch filter set [slot]")
        print("       kbch filter [option]")
        print("\nOptions:")
        print("  --list, -l   Show available slots")
        print("  --help, -h   Show this help message")

    def show_available_slots():
        print("\nAvailable slots:")
        for i in range(1, 7):
            print(f"  {i}")

    # No command --------------------------------------------------------------

    if len(args) < 1:
        print("❌ Error: No filter slot provided.")
        print("ℹ️ Use 'kbch filter --help' for usage information.")
        sys.exit(1)

    # Help --------------------------------------------------------------------

    if args[0] in ['--help', '-h']:
        show_help()
        sys.exit(0)

    # List --------------------------------------------------------------------

    if args[0] in ['--list', '-l']:
        show_available_slots()
        sys.exit(0)

    # Set ---------------------------------------------------------------------

    if args[0] in ['set']:
        if len(args) < 2:
            print("❌ Error: No slot provided.")
            show_available_slots()
            sys.exit(1)
        try:
            slot = int(args[1])
        except ValueError:
            print("❌ Error: Slot must be an integer between 1 and 6.")
            show_available_slots()
            sys.exit(1)
        if slot < 1 or slot > 6:
            print("❌ Error: Invalid slot number.")
            show_available_slots()
            sys.exit(1)

        print(f'⌛ Setting filter wheel to slot {slot}...')
        if is_config_set():
            config = get_config()
            fw = kbench.FilterWheel(port=config['filter']['port'])
        else:
            fw = kbench.FilterWheel()
        fw.move(slot)
        print("✅ Done")
        sys.exit(0)

    # Invalid args ------------------------------------------------------------
    print(f"❌ Error: Invalid filter command.")
    print("ℹ️ Use 'kbch filter --help' for usage information.")
    sys.exit(1)

#==============================================================================
# EOF
#==============================================================================

if __name__ == "__main__":
    main()
    sys.exit(0)