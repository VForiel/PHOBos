# Automatic mode detection (control or sandbox)
try:
    import bmc
    SANDBOX_MODE = False
    print("✅ BMC lib found. Running in control mode.")
except ImportError:
    from .sandbox import bmc_mock as bmc
    SANDBOX_MODE = True
    print("❌ BMC lib not found. Install it via the BMC SDK.")
    print("⛱️ Running in sandbox mode.")

# Import classes
from .classes import PupilMask, FilterWheel, DM

# Make bmc, SANDBOX_MODE and classes available for other modules
__all__ = ['bmc', 'SANDBOX_MODE', 'PupilMask', 'FilterWheel', 'DM']