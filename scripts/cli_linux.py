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
    print("📋 KBENCH - Ker... uh, Photonics Bench Control Interface")
    print("="*50)
    print("Usage: kbch [equipment] [command] [options]")
    print("       kbch config [command] [options]")
    print("       kbch [global-options]")
    
    print("\n🔧 Available Equipment:")
    print("  mask     Control pupil mask (rotation + positioning)")
    print("  filter   Control filter wheel (slot selection)")
    print("  config   Manage configuration files and settings")
    
    print("\n⚙️  Global Options:")
    print("  -h, --help     Show this help message and exit")
    print("  -v, --version  Show version information and exit")
    
    print("\n💡 Examples:")
    print("  kbch mask set 3           # Rotate mask to 180° (3×60°)")
    print("  kbch mask get             # Show current mask position")
    print("  kbch filter set 2         # Move filter wheel to slot 2")
    print("  kbch filter get           # Show current filter position")
    print("  kbch config create.yml    # Create new configuration")
    
    print("\n📖 For detailed help on specific equipment:")
    print("  kbch [equipment] --help   # e.g., kbch mask --help")

def show_version():
    # Get the version from the ../pyproject.toml file
    try:
        import toml
        pyproject_file = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
        with open(pyproject_file, 'r') as f:
            pyproject = toml.load(f)
        version = pyproject['project']['version']
        print(f"ℹ️ kbch script version: {version}")
        print(f"ℹ️ kbench module version: {kbench.__version__}")
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
        print("⚙️  CONFIGURATION - Settings Management")
        print("="*45)
        print("Usage: kbch config [path]")
        print("       kbch config create [path]")
        print("       kbch config [command] [options]")
        print("       kbch config [equipment] [action] [name]")
        
        print("\n📁 File Management:")
        print("  [path]         Set active configuration file (.yml/.json)")
        print("  create [path]  Create new configuration interactively")
        
        print("\n📋 Information:")
        print("  -s, --show     Show current configuration file path")
        print("  -h, --help     Show this help message")
        print("  -r, --reset    Reset to default (no configuration)")
        
        print("\n📐 Mask Management:")
        print("  mask add [name]     Save current mask position with name")
        print("  mask remove [name]  Remove saved mask configuration")
        print("  mask list           Show all configured mask positions")
        
        print("\n🔍 Filter Management:")
        print("  filter add [name]     Save current filter position with name")
        print("  filter remove [name]  Remove saved filter configuration")
        print("  filter list           Show all configured filter positions")
        
        print("\n💡 Examples:")
        print("  kbch config create my_setup.yml    # Interactive setup")
        print("  kbch config my_setup.yml           # Use configuration")
        print("  kbch config mask add \"center\"       # Save current position")

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

    # Create ------------------------------------------------------------------

    if args[0] in ['create']:
        if len(args) < 2:
            print("❌ Error: No config file path provided.")
            print("ℹ️ Usage: kbch config create [path]")
            sys.exit(1)
        
        config_file_path = args[1]
        
        # Check if path ends with .yml or .yaml
        if not (config_file_path.endswith('.yml') or config_file_path.endswith('.yaml')):
            print("❌ Error: Configuration file must have .yml or .yaml extension.")
            sys.exit(1)
        
        # Check if file already exists
        if os.path.exists(config_file_path):
            response = input(f"⚠️  File '{config_file_path}' already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("❌ Operation cancelled.")
                sys.exit(1)
        
        print("🔧 Creating new configuration file...")
        print("📝 Please provide the following information:")
        
        # Get mask configuration
        print("\n📐 Mask Configuration:")
        newport_port = input("  Newport port (default: /dev/ttyUSB0): ").strip() or "/dev/ttyUSB0"
        zaber_port = input("  Zaber port (default: /dev/ttyUSB1): ").strip() or "/dev/ttyUSB1"
        
        # Get filter configuration
        print("\n🔍 Filter Configuration:")
        filter_port = input("  Filter wheel port (default: /dev/ttyUSB2): ").strip() or "/dev/ttyUSB2"
        
        # Create configuration structure
        config = {
            'mask': {
                'ports': {
                    'newport': newport_port,
                    'zaber': zaber_port
                },
                'slots': {}
            },
            'filter': {
                'port': filter_port,
                'slots': {}
            }
        }
        
        # Write configuration file
        try:
            with open(config_file_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
            
            print(f"✅ Configuration file created: {config_file_path}")
            
            # Ask if user wants to set it as active config
            response = input("🔗 Set this as the active configuration? (Y/n): ")
            if response.lower() != 'n':
                CONFIG['config_path'] = os.path.abspath(config_file_path)
                with open(CONFIG_FILE, "w") as f:
                    json.dump(CONFIG, f, indent=4)
                print("✅ Configuration set as active.")
            
        except Exception as e:
            print(f"❌ Error creating configuration file: {e}")
            sys.exit(1)
        
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
        # get_pos() returns (wheel_angle, zaber_vertical, zaber_horizontal)
        wheel_pos, zab_v_pos, zab_h_pos = p.get_pos()
        x_pos = zab_h_pos  # horizontal position in steps
        y_pos = zab_v_pos  # vertical position in steps  
        a_pos = wheel_pos  # wheel angle in degrees
        
        # Update config
        if 'mask' not in config:
            config['mask'] = {'slots': {}, 'ports': {'newport': '/dev/ttyUSB0', 'zaber': '/dev/ttyUSB1'}}
        if 'slots' not in config['mask']:
            config['mask']['slots'] = {}
            
        config['mask']['slots'][mask_name] = {
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
        fw = kbench.FilterWheel(filter_port=config.get('filter', {}).get('port', '/dev/ttyUSB0'))
        
        # Get current position
        current_slot = fw.get_pos()
        
        # Update config
        if 'filter' not in config:
            config['filter'] = {'slots': {}, 'port': '/dev/ttyUSB0'}
        if 'slots' not in config['filter']:
            config['filter']['slots'] = {}
            
        config['filter']['slots'][filter_name] = {
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
        print("📐 MASK CONTROL - Pupil Mask Management")
        print("="*45)
        print("Usage: kbch mask get")
        print("Usage: kbch mask set [mask]")
        print("Usage: kbch mask home")
        print("       kbch mask move <axis> [value] [options]")
        print("       kbch mask [options]")
        
        print("\n🎯 Set Commands:")
        print("  set [mask]     Apply mask position (name or number 1-6)")
        
        print("\n🔄 Movement Commands:")
        print("  mvh [value]    Move horizontally (X-axis) in steps")
        print("  mvv [value]    Move vertically (Y-axis) in steps")
        print("  mva [degrees]  Rotate wheel by angle in degrees")
        
        print("\n📋 Information:")
        print("  -l, --list     Show all available masks")
        print("  -h, --help     Show this help message")
        
        show_available_masks()
        
        print("\n⚡ Movement Options:")
        print("  -a, --abs      Use absolute positioning (default: relative)")
        
        print("\n💡 Quick Rotation (bypasses config):")
        print("  Numbers 1-6 rotate directly to n×60° without moving X/Y axes")
        print("  Example: 'kbch mask set 3' → 180° rotation only")

    def show_available_masks():
        print("\n🎭 Available Masks:")
        if config and masks:
            print("  📝 Configured masks:")
            for name in masks.keys():
                if isinstance(name, str):  # Named masks from config
                    print(f"    {name}")
            print("  🔢 Quick rotation (bypasses config):")
            for i in range(1, 7):
                print(f"    {i} → {i*60}° rotation")
        else:
            print("  🔢 Default positions (no config file):")
            for i in range(1, 7):
                print(f"    {i} → {i*60}° rotation (X=0, Y=0)")

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

    # Home --------------------------------------------------------------------

    if args[0] in ['home']:
        print("⌛ Homing mask...")
        
        p = kbench.PupilMask(
            newport_port=config['mask']['ports']['newport'] if config else '/dev/ttyUSB0',
            zaber_port=config['mask']['ports']['zaber'] if config else '/dev/ttyUSB1'
        )
        p.newport.home_search()
        
        print("✅ Done")
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
                print(f'⌛ Setting mask to position {mask_number}...')
                
                p = kbench.PupilMask(
                    newport_port=config['mask']['ports']['newport'] if config else '/dev/ttyUSB0',
                    zaber_port=config['mask']['ports']['zaber'] if config else '/dev/ttyUSB1'
                )
                
                # Only rotate the wheel, don't touch x and y axes
                p.apply_mask(mask_number)
                
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

    # Get position ------------------------------------------------------------

    if args[0] in ['get']:
        print("⌛ Getting current mask position...")
        
        p = kbench.PupilMask(
            newport_port=config['mask']['ports']['newport'] if config else '/dev/ttyUSB0',
            zaber_port=config['mask']['ports']['zaber'] if config else '/dev/ttyUSB1'
        )
        
        # Get current positions
        # get_pos() returns (wheel_angle, zaber_vertical, zaber_horizontal)
        wheel_pos, zab_v_pos, zab_h_pos = p.get_pos()
        
        print(f"📐 Current mask position:")
        print(f"  🔄 Wheel angle: {wheel_pos}°")
        print(f"  ↔️  Horizontal (X): {zab_h_pos} steps")
        print(f"  ↕️  Vertical (Y): {zab_v_pos} steps")
        
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
        print("🔍 FILTER CONTROL - Filter Wheel Management")
        print("="*45)
        print("       kbch filter get")
        print("Usage: kbch filter set [slot|name]")
        print("       kbch filter [options]")
        
        print("\n🎯 Commands:")
        print("  set [target]   Move to slot number (1-6) or configured filter name")
        
        print("\n📋 Information:")
        print("  -l, --list     Show available slots and configured filters")
        print("  -h, --help     Show this help message")
        
        print("\n💡 Examples:")
        print("  kbch filter set 3           # Move to slot 3")
        print("  kbch filter set \"ND_filter\"  # Use configured filter name")

    def show_available_slots():
        print("\n🎰 Available Options:")
        print("  🔢 Slot numbers:")
        for i in range(1, 7):
            print(f"    {i}")
        
        if filters:
            print("  📝 Configured filters:")
            for name, filter_config in filters.items():
                slot = filter_config.get('slot', 'N/A')
                print(f"    {name} (→ slot {slot})")

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

    # Get position ------------------------------------------------------------

    if args[0] in ['get']:
        print("⌛ Getting current filter position...")
        
        if is_config_set():
            config = get_config()
            fw = kbench.FilterWheel(filter_port=config.get('filter', {}).get('port', '/dev/ttyUSB0'))
        else:
            fw = kbench.FilterWheel()
        
        current_slot = fw.get_pos()
        
        print(f"🔍 Current filter position: slot {current_slot}")
        
        # If there are configured filters, show which one matches
        if filters:
            matching_filters = [name for name, filter_config in filters.items() 
                              if filter_config.get('slot') == current_slot]
            if matching_filters:
                print(f"  📝 Configured filter(s): {', '.join(matching_filters)}")
        
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
            fw = kbench.FilterWheel(filter_port=config.get('filter', {}).get('port', '/dev/ttyUSB0'))
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