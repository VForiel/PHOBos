# Automatic mode detection (control or sandbox)
try:
    import bmc
    print("✅ BMC lib found. Running in control mode.")
    SANDBOX_MODE = False
except ImportError:
    from .sandbox import bmc_mock as bmc
    print("❌ BMC lib not found. Install it via the BMC SDK.")
    print("⛱️ Running in sandbox mode.")
    SANDBOX_MODE = True

# Serial library selection
if SANDBOX_MODE:
    from .sandbox import serial_mock as serial
else:
    import serial

# Import classes
from .classes import PupilMask, FilterWheel, DM

# Make bmc, serial and classes available for other modules
__all__ = ['bmc', 'serial', 'PupilMask', 'FilterWheel', 'DM']