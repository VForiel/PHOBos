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

# Make bmc and SANDBOX_MODE available for other modules
__all__ = ['bmc', 'SANDBOX_MODE']

from .classes import *
from .modules import *