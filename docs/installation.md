# Lab PC Setup Guide

This guide provides installation and setup instructions for the PHOTONICS bench laboratory computer.

## Camera

Camera setup documentation to be added.

## Deformable Mirror (BMC)

### Software Download

The BMC SDK can be downloaded directly from the Boston Micromachines website:

[https://bostonmicromachines.com/DMSDK/BMC-DMSDK.zip](https://bostonmicromachines.com/DMSDK/BMC-DMSDK.zip)

:::{note}
The SDK requirements tie us to a specific version of Ubuntu Linux.
:::

After successful installation, files will be added to the `/opt/Boston Micromachines/` directory.

**Verify installation** by checking that the Linux kernel module is loaded:

```bash
lsmod | grep bmc
```

Expected output:
```
bmc_mdrv         16384  0
```

### Configuration Files

Each BMC DM is identified by an 11-digit serial number.

**PHOTONICS bench DM serial number:** `27BW007#051`

Two configuration files must be copied to the appropriate locations:

```bash
sudo cp 27BW007#051.dm /opt/Boston\ Micromachines/Profiles/
sudo cp LUT_27BW007#051.mat /opt/Boston\ Micromachines/Calibration/
```

:::{note}
The `.dm` profile file is mandatory. The `.mat` calibration file may only be used by MATLAB-based tools.
:::

These files are located in the `config/calibration/` directory of this repository.

### Testing the Installation

BMC provides command-line test tools in `/opt/Boston Micromachines/bin/`.

**Test DM communication:**

```bash
/opt/Boston\ Micromachines/bin/BmcExampleC -s 27BW007#051
```

This runs through a sequence of actuator activations to verify proper operation.

## Python Environment Setup

### Lab PC (Linux)

Install all dependencies including hardware-specific libraries:

```bash
# Install base package with lab dependencies
pip install -r requirements-lab.txt
pip install -e .
```

This installs:
- Core dependencies (numpy, scipy, etc.)
- xaosim (for camera shared memory interface)
- Development tools (pytest, jupyter, etc.)

### Development/Sandbox Mode (Windows/Linux without hardware)

For development without hardware access:

```bash
pip install -e .
```

This installs only the base dependencies. The package will automatically run in **sandbox mode** with mock hardware interfaces.

## Verifying Installation

Test the installation in Python:

```python
import kbench

# Check mode
print(f"Sandbox mode: {kbench.SANDBOX_MODE}")

# Try initializing hardware (uses mocks if in sandbox mode)
dm = kbench.DM()
fw = kbench.FilterWheel()
chip = kbench.Chip(6)
camera = kbench.Cred3()
```

If BMC SDK is not installed, you'll see:
```
❌ BMC lib not found. Install it via the BMC SDK.
⛱️ Running in sandbox mode.
```

## Additional Resources

- {doc}`deformable_mirror` - DM control API documentation
- {doc}`cred3` - Camera interface documentation
- {doc}`configuration_file` - Configuration file format
