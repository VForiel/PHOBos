# Copilot Instructions for Kbench-Controls

## Architecture Overview

**Kbench-Controls** is a Python package for controlling optical test bench equipment in a kernel-nulling interferometry setup. The system manages three main hardware components: deformable mirrors (DM), filter wheels, and pupil masks with motion controllers.

### Core Architecture Patterns

**Sandbox Mode Strategy**: The system implements a dual-mode architecture for development/testing without hardware:
- `SANDBOX_MODE = True` when BMC library is unavailable → uses mock classes from `src/kbench/sandbox/`
- `SANDBOX_MODE = False` for production → uses real hardware drivers
- All output prefixed with `⛱️ [SANDBOX]` in mock mode

**Hardware Classes**: Located in `src/kbench/classes/`, each inherits common patterns:
- `DM` (deformable_mirror.py): 169-segment hexagonal mirror control via BMC SDK
- `FilterWheel` (filter_wheel.py): 6-position Thorlabs wheel via serial
- `PupilMask` (pupil_mask.py): Combined Newport rotary + dual Zaber linear stages

**Configuration System**: 
- YAML/JSON config files for hardware ports and calibration data
- User config in `~/.kbch.json`, device configs like `DM_config.json`
- CLI tool `kbch` for interactive hardware control

## Development Workflow

### Environment Setup
```bash
# Conda is the preferred package manager for scientific environments
conda create -n kbench python=3.12
conda activate kbench
# Install package in development mode
pip install -e .
```

### Working in Sandbox Mode
When BMC SDK is not installed, the system automatically enters sandbox mode. All mock classes print debug information and simulate hardware responses. This is the default for development.

### Hardware Integration Points
- **BMC SDK**: Proprietary deformable mirror control (Windows/Linux)
- **Serial Communications**: PySerial for filter wheel and motion controllers
- **Config Loading**: PyYAML for configuration management

### Critical Dependencies
- `numpy`: Array operations for 169-segment DM control
- `pyserial`: All motor and filter communications
- `pyyaml`: Configuration file parsing

## Project-Specific Conventions

### Error Handling
- Hardware availability checked at import time in `__init__.py`
- Graceful degradation to sandbox mode with clear user feedback
- Serial timeouts and connection validation in all hardware classes

### Configuration Management
- Use absolute paths for device ports (e.g., `/dev/ttyUSB0`)
- Segment arrays are 0-indexed (0-168 for 169 segments)
- Position coordinates: piston (nm), tip/tilt (radians)

### CLI Design Patterns
```bash
kbch mask set 3           # Set mask position
kbch filter get           # Query current state  
kbch config mask add "name"  # Save current position
```

## Key Files for Understanding

- `src/kbench/__init__.py`: Mode detection and main imports
- `src/kbench/classes/deformable_mirror.py`: 169-segment control patterns
- `scripts/cli_linux.py`: Complete CLI interface implementation
- `DM_config.json`: Example of 169-segment calibration data
- `config.yml`: Hardware port mapping template

## Testing and Validation

Run in sandbox mode by default (no hardware required). For hardware testing, ensure:
1. BMC SDK installed for deformable mirror
2. Correct serial port permissions and device connections
3. Configuration files match actual hardware setup

The `timeit_tests.ipynb` notebook contains performance benchmarks for segment update operations.

## Common Debugging Steps

1. Check `SANDBOX_MODE` status at startup
2. Verify serial port permissions: `ls -la /dev/ttyUSB*`
3. Test configuration loading: `kbch config --show`
4. Monitor sandbox output for command simulation