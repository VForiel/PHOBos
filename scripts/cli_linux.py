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

    # Filter wheel ------------------------------------------------------------

    if sys.argv[1] in ['filter']:   
        control_filter(sys.argv[2:])
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
    print("  --version, -v  Show version information")

def show_version():
    # Get the version from the ../pyproject.toml file
    try:
        import toml
        pyproject_file = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
        with open(pyproject_file, 'r') as f:
            pyproject = toml.load(f)
        version = pyproject['project']['version']
        print(f"ℹ️ kbench version {version}")
    except Exception as e:
        print("❌ Error: Could not retrieve version information.")
        print(f"ℹ️ {e}")
        sys.exit(1)

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
        print("       kbch config mask [add|remove|list] [name]")
        print("       kbch config filter [add|remove|list] [name]")
        print("\nOptions:")
        print("  --help, -h     Show this help message")
        print("  --reset, -r    Reset configuration to default")
        print("  --show, -s     Show current configuration")
        print("\nMask/Filter management:")
        print("  mask add [name]     Add current mask position to config")
        print("  mask remove [name]  Remove mask from config")
        print("  mask list           List all configured masks")
        print("  filter add [name]   Add current filter position to config")
        print("  filter remove [name] Remove filter from config")
        print("  filter list         List all configured filters")

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

    if os.path.exists(args[0]) and (args[0].endswith('.json') or args[0].endswith('.yml') or args[0].endswith('.yaml')):
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

    # Mask management ---------------------------------------------------------

    if args[0] in ['mask']:
        if not is_config_set():
            print("❌ Error: No configuration file set. Please set a config file first.")
            sys.exit(1)
        
        control_mask_config(args[1:])
        sys.exit(0)

    # Filter management ------------------------------------------------------

    if args[0] in ['filter']:
        if not is_config_set():
            print("❌ Error: No configuration file set. Please set a config file first.")
            sys.exit(1)
        
        control_filter_config(args[1:])
        sys.exit(0)

    # Invalid args ------------------------------------------------------------

    print(f"❌ Error: Invalid config path. The path should point to a .json, .yml, or .yaml file.")
    print("ℹ️ Use 'kbch config --help' for usage information.")
    sys.exit(1)

#==============================================================================
# Config Mask Management
#==============================================================================

def control_mask_config(args):
    
    if len(args) < 1:
        print("❌ Error: No mask command provided.")
        print("ℹ️ Use 'kbch config --help' for usage information.")
        sys.exit(1)
    
    config_path = get_config_file_path()
    
    # Add mask ----------------------------------------------------------------
    
    if args[0] in ['add']:
        if len(args) < 2:
            print("❌ Error: No mask name provided.")
            print("ℹ️ Usage: kbch config mask add [name]")
            sys.exit(1)
        
        mask_name = args[1]
        
        # Get current positions
        print("⌛ Reading current mask positions...")
        config = get_config()
        p = kbench.PupilMask(
            newport_port=config['mask']['ports']['newport'],
            zaber_port=config['mask']['ports']['zaber']
        )
        
        # Get current positions
        x_pos = p.get_position_h()
        y_pos = p.get_position_v()
        a_pos = p.get_position_a()
        
        # Update config
        if 'mask' not in config:
            config['mask'] = {'slots': {}, 'ports': {'newport': '/dev/ttyUSB0', 'zaber': '/dev/ttyUSB1'}}
        if 'slots' not in config['mask']:
            config['mask']['slots'] = {}
            
        config['mask']['slots'][mask_name] = {
            'name': mask_name,
            'x': x_pos,
            'y': y_pos,
            'a': a_pos
        }
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Mask '{mask_name}' added with positions: x={x_pos}, y={y_pos}, a={a_pos}°")
        sys.exit(0)
    
    # Remove mask -------------------------------------------------------------
    
    if args[0] in ['remove']:
        if len(args) < 2:
            print("❌ Error: No mask name provided.")
            print("ℹ️ Usage: kbch config mask remove [name]")
            sys.exit(1)
        
        mask_name = args[1]
        config = get_config()
        
        if 'mask' not in config or 'slots' not in config['mask'] or mask_name not in config['mask']['slots']:
            print(f"❌ Error: Mask '{mask_name}' not found in configuration.")
            sys.exit(1)
        
        del config['mask']['slots'][mask_name]
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Mask '{mask_name}' removed from configuration.")
        sys.exit(0)
    
    # List masks --------------------------------------------------------------
    
    if args[0] in ['list']:
        config = get_config()
        
        if 'mask' not in config or 'slots' not in config['mask'] or not config['mask']['slots']:
            print("📋 No masks configured.")
            sys.exit(0)
        
        print("📋 Configured masks:")
        for name, mask_config in config['mask']['slots'].items():
            x = mask_config.get('x', 'N/A')
            y = mask_config.get('y', 'N/A')
            a = mask_config.get('a', 'N/A')
            print(f"  {name}: x={x}, y={y}, a={a}°")
        sys.exit(0)
    
    print("❌ Error: Invalid mask config command.")
    print("ℹ️ Use 'kbch config --help' for usage information.")
    sys.exit(1)

#==============================================================================
# Config Filter Management
#==============================================================================

def control_filter_config(args):
    
    if len(args) < 1:
        print("❌ Error: No filter command provided.")
        print("ℹ️ Use 'kbch config --help' for usage information.")
        sys.exit(1)
    
    config_path = get_config_file_path()
    
    # Add filter --------------------------------------------------------------
    
    if args[0] in ['add']:
        if len(args) < 2:
            print("❌ Error: No filter name provided.")
            print("ℹ️ Usage: kbch config filter add [name]")
            sys.exit(1)
        
        filter_name = args[1]
        
        # Get current position
        print("⌛ Reading current filter position...")
        config = get_config()
        fw = kbench.FilterWheel(port=config.get('filter', {}).get('port', '/dev/ttyUSB0'))
        
        # Get current position
        current_slot = fw.get_position()
        
        # Update config
        if 'filter' not in config:
            config['filter'] = {'slots': {}, 'port': '/dev/ttyUSB0'}
        if 'slots' not in config['filter']:
            config['filter']['slots'] = {}
            
        config['filter']['slots'][filter_name] = {
            'name': filter_name,
            'slot': current_slot
        }
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Filter '{filter_name}' added at slot {current_slot}")
        sys.exit(0)
    
    # Remove filter -----------------------------------------------------------
    
    if args[0] in ['remove']:
        if len(args) < 2:
            print("❌ Error: No filter name provided.")
            print("ℹ️ Usage: kbch config filter remove [name]")
            sys.exit(1)
        
        filter_name = args[1]
        config = get_config()
        
        if 'filter' not in config or 'slots' not in config['filter'] or filter_name not in config['filter']['slots']:
            print(f"❌ Error: Filter '{filter_name}' not found in configuration.")
            sys.exit(1)
        
        del config['filter']['slots'][filter_name]
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Filter '{filter_name}' removed from configuration.")
        sys.exit(0)
    
    # List filters ------------------------------------------------------------
    
    if args[0] in ['list']:
        config = get_config()
        
        if 'filter' not in config or 'slots' not in config['filter'] or not config['filter']['slots']:
            print("📋 No filters configured.")
            sys.exit(0)
        
        print("📋 Configured filters:")
        for name, filter_config in config['filter']['slots'].items():
            slot = filter_config.get('slot', 'N/A')
            print(f"  {name}: slot={slot}")
        sys.exit(0)
    
    print("❌ Error: Invalid filter config command.")
    print("ℹ️ Use 'kbch config --help' for usage information.")
    sys.exit(1)

#==============================================================================
# Control mask wheel
#==============================================================================

def control_mask(args):

    
    config = None
    masks = {}
    
    if is_config_set():
        config = get_config()
        # Load masks from config - using names as keys
        if 'mask' in config and 'slots' in config['mask']:
            for name, mask_config in config['mask']['slots'].items():
                masks[name] = mask_config
    else:
        # Default masks when no config is set
        masks = {
            1:{
                'x': 0,
                'y': 0,
                'a': 0,
            },
            2:{
                'x': 0,
                'y': 0,
                'a': 60,
            },
            3:{
                'x': 0,
                'y': 0,
                'a': 120,
            },
            4:{
                'x': 0,
                'y': 0,
                'a': 180,
            },
            5:{
                'x': 0,
                'y': 0,
                'a': 240,
            },
            6:{
                'x': 0,
                'y': 0,
                'a': 300,
            },
        }

    def show_help():
        print("Usage: kbch mask set [mask]")
        print("       kbch mask mvh [-a|--abs] [value (int)]")
        print("       kbch mask mvv [-a|--abs] [value (int)]")
        print("       kbench mask mva [-a|--abs] [value (float)]")
        print("       kbch mask [option]")
        show_available_masks()
        print("\\nNote: You can use numbers 1-6 to directly rotate the wheel to n*60° without")
        print("      affecting x/y axes (overrides configuration file settings)")
        print("\\nOptions:")
        print("  --list, -l   Show available masks")
        print("  --help, -h   Show this help message")

    def show_available_masks():
        print("\\nAvailable masks:")
        for name in masks.keys():
            print(f"  {name}")
        if config:
            print("\\nDirect rotation (1-6): Bypasses config, rotates to n*60°")

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

        # Check if the mask argument is a number from 1 to 6
        try:
            mask_number = int(args[1])
            if 1 <= mask_number <= 6:
                # Use direct rotation without configuration file override
                print(f'⌛ Setting mask to position {mask_number} (rotation: {mask_number * 60}°)...')
                
                p = kbench.PupilMask(
                    newport_port=config['mask']['ports']['newport'] if config else '/dev/ttyUSB0',
                    zaber_port=config['mask']['ports']['zaber'] if config else '/dev/ttyUSB1'
                )
                
                # Only rotate the wheel, don't touch x and y axes
                p.rotate(mask_number * 60, abs=True)
                
                print("✅ Done")
                sys.exit(0)
        except ValueError:
            # Not a number, continue with normal mask name logic
            pass

        if args[1] in masks:
            mask = args[1]
        else:
            print("❌ Error: Invalid mask value.")
            show_available_masks()
            sys.exit(1)

        print(f'⌛ Setting "{mask}" mask...')
        
        p = kbench.PupilMask(
            newport_port=config['mask']['ports']['newport'] if config else '/dev/ttyUSB0',
            zaber_port=config['mask']['ports']['zaber'] if config else '/dev/ttyUSB1'
        )

        # Use the mask configuration (either from config file or defaults)
        mask_config = masks[mask]
        p.rotate(mask_config['a'], abs=True)
        p.move_h(mask_config['x'], abs=True)
        p.move_v(mask_config['y'], abs=True)

        print("✅ Done")
        sys.exit(0)

    # Move --------------------------------------------------------------------

    if args[0] in ['mvh', 'mvv', 'mva']:

        print(f"⌛ Moving mask...")

        p = kbench.PupilMask(
            newport_port=config['mask']['ports']['newport'] if config else '/dev/ttyUSB0',
            zaber_port=config['mask']['ports']['zaber'] if config else '/dev/ttyUSB1'
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
            p.rotate(value, abs=abs)

        print("✅ Done")
        sys.exit(0)

#==============================================================================
# Control filter wheel
#==============================================================================

def control_filter(args):
    
    config = None
    filters = {}
    
    if is_config_set():
        config = get_config()
        # Load filters from config - using names as keys
        if 'filter' in config and 'slots' in config['filter']:
            for name, filter_config in config['filter']['slots'].items():
                filters[name] = filter_config
    
    def show_help():
        print("Usage: kbch filter set [slot|name]")
        print("       kbch filter [option]")
        print("\\nOptions:")
        print("  --list, -l   Show available slots and configured filters")
        print("  --help, -h   Show this help message")

    def show_available_slots():
        print("\\nAvailable slots:")
        for i in range(1, 7):
            print(f"  {i}")
        
        if filters:
            print("\\nConfigured filters:")
            for name, filter_config in filters.items():
                slot = filter_config.get('slot', 'N/A')
                print(f"  {name} (slot {slot})")

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
            print("❌ Error: No slot or filter name provided.")
            show_available_slots()
            sys.exit(1)
        
        # Check if it's a configured filter name
        if args[1] in filters:
            filter_name = args[1]
            slot = filters[filter_name]['slot']
            print(f'⌛ Setting filter "{filter_name}" (slot {slot})...')
        else:
            # Try to parse as slot number
            try:
                slot = int(args[1])
                if slot < 1 or slot > 6:
                    print("❌ Error: Invalid slot number.")
                    show_available_slots()
                    sys.exit(1)
                print(f'⌛ Setting filter wheel to slot {slot}...')
            except ValueError:
                print("❌ Error: Invalid filter name or slot number.")
                show_available_slots()
                sys.exit(1)

        if is_config_set():
            config = get_config()
            fw = kbench.FilterWheel(port=config.get('filter', {}).get('port', '/dev/ttyUSB0'))
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