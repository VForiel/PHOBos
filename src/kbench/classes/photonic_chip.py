import numpy as np
from .. import serial
import time
import matplotlib.pyplot as plt
from .. import SANDBOX_MODE
import re
import warnings


class XPOW:
    """
    Singleton class managing the serial connection to the XPOW controller. You can simply use `kbench.xpow` to access it.
    
    This class ensures a single shared connection is used by all Chip and Channel instances.
    It is automatically instantiated on first access and handles all low-level communication.
    
    Attributes
    ----------
    N_CHANNELS : int
        Total number of channels available (40).
    MAX_VOLTAGE : float
        Maximum voltage in V (5V).
    MAX_CURRENT : float
        Maximum current in mA (300mA).
    CUR_CONVERSION : float
        Conversion factor from mA to DAC units (65535/300).
    VOLT_CONVERSION : float
        Conversion factor from V to DAC units (65535/40).
    CUR_CORRECTION : np.ndarray
        Per-channel calibration multipliers for current (initialized to 1.0).
    VOLT_CORRECTION : np.ndarray
        Per-channel calibration multipliers for voltage (initialized to 1.0).

    Access
    ------
    The singleton instance is available as `kbench.xpow`, `Arch.xpow`, or `PhaseShifter.xpow`.
    """
    
    _instance = None
    _serial = None
    
    # Hardware specifications
    N_CHANNELS = 40

    # Conversion factors (fixed, hardware-dependent)
    # To convert user values (mA, V) to 16-bit DAC values
    CUR_CONVERSION = 65535 / 300  # DAC units per mA
    VOLT_CONVERSION = 65535 / 40  # DAC units per V

    # Securities
    MAX_VOLTAGE = 30  # V
    MAX_CURRENT = 300  # mA
    
    # Correction coefficients (calibrable, initialized to 1.0)
    # Refined by update_coeffs() to compensate for channel variations
    CUR_CORRECTION = np.ones(N_CHANNELS)
    VOLT_CORRECTION = np.ones(N_CHANNELS)
    
    # Phase-to-voltage conversion coefficients (radians to V)
    # PHASE_CONVERSION[ch] gives the voltage needed per radian of phase shift
    # Default: 2π phase shift at 0.6W with I=300mA → V=2V → 2V/(2π) ≈ 0.318 V/rad
    PHASE_CONVERSION = np.ones(N_CHANNELS) * (2.0 / (2 * np.pi))
    
    def __new__(cls):
        """Singleton pattern: return existing instance or create new one."""
        if cls._instance is None:
            cls._instance = super(XPOW, cls).__new__(cls)
        return cls._instance
    
    def connect(self):
        """
        Establish serial connection to the XPOW controller.
        
        Returns
        -------
        serial.Serial
            Active serial connection object.
            
        Raises
        ------
        ConnectionError
            If the XPOW controller does not respond correctly.
            
        Notes
        -----
        - In sandbox mode, uses '/dev/ttyACM0' as default port
        - In normal mode, auto-detects XPOW via USB VID:PID (2341:8036)
        - Baudrate: 115200 baud
        - Timeout: 1.0 second
        """
        if self._serial is None:
            # Setup serial connection
            if SANDBOX_MODE:
                port = '/dev/ttyACM0'
            else:
                from serial.tools import list_ports
                port = list(list_ports.grep("2341:8036"))[0][0]   
            self._serial = serial.Serial(port, baudrate=115200, timeout=1.0)
            
            # Check connection
            res = self.send_command("*IDN?")
            if "XPOW" not in res:
                raise ConnectionError(f"❌ No response from the XPOW controller on port {port} at 115200 bauds. Response was: {res}")
        return self._serial
    
    def disconnect(self):
        """
        Close the serial connection to the XPOW controller.
        
        Notes
        -----
        This method is automatically called when the program exits, but can be
        called manually to release the serial port.
        """
        if self._serial is not None:
            self._serial.close()
            self._serial = None
    
    def send_command(self, cmd: str, verbose: bool = False, output: bool = True) -> str:
        """
        Send a command to the XPOW controller and return the response.
        
        Parameters
        ----------
        cmd : str
            Command string to send (without newline terminator).
        verbose : bool, optional
            If True, print transmitted and received messages. Default is False.
        output : bool, optional
            If True, wait for and return response. Default is True.
            
        Returns
        -------
        str or None
            Response from XPOW controller if output=True, None otherwise.
            
        Notes
        -----
        Common XPOW commands:
        
        - ``*IDN?`` : Query device identification
        - ``CH:X:CUR:Y`` : Set current Y on channel X
        - ``CH:X:VOLT:Y`` : Set voltage Y on channel X
        - ``CH:X:VAL?`` : Query voltage and current on channel X
        """
        cmd_line = cmd + "\n"
        self.connect()
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        if verbose:
            print(f"📤 XPOW TX: '{cmd}'")
        self._serial.write(cmd_line.encode())
        time.sleep(0.01)  # Wait a bit for the command to be processed
        if output:
            response = self._serial.readline().decode().strip()
            if verbose:
                print(f"📥 XPOW RX: '{response}'")
            return response
        else:
            if verbose:
                print(f"📥 Output disabled")
            return None
    
    @staticmethod
    def update_all_coeffs(plot: bool = False, verbose: bool = False):
        """
        Calibrate correction coefficients for ALL 40 XPOW channels.
        
        Scans current and voltage for every channel, queries measured values,
        and refines CUR_CORRECTION and VOLT_CORRECTION for all channels.
        
        Parameters
        ----------
        plot : bool, optional
            If True, display fit results for visual inspection. Default is False.
        verbose : bool, optional
            If True, print calibration details. Default is False.
            
        Notes
        -----
        This method updates all 40 channels. For calibrating only specific chip
        channels, use Arch.update_coeffs() instead. For a single channel, use
        PhaseShifter.update_coeff().
        
        The calibration works by:
        1. Setting known currents/voltages on each channel
        2. Measuring actual output values
        3. Computing slope of (set vs measured)
        4. Updating CORRECTION[ch] = old_CORRECTION[ch] / slope
        """
        if verbose:
            print(f"🔧 Calibrating all {XPOW.N_CHANNELS} XPOW channels...")
        
        for ch in range(1, XPOW.N_CHANNELS + 1):
            channel = PhaseShifter(ch)
            channel.update_coeff(plot=plot, verbose=verbose)
        
        print("✅ All XPOW correction coefficients updated.")
    
    @staticmethod
    def turn_off(verbose: bool = False):
        """
        Set voltage and current to zero on ALL 40 XPOW channels.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print shutdown commands. Default is False.
            
        Notes
        -----
        This is a global safety method affecting all 40 channels simultaneously,
        regardless of which chip architectures are in use.
        
        Examples
        --------
        >>> XPOW.turn_off()  # Turn off all channels
        """
        _xpow.send_command(f"CH:1-{XPOW.N_CHANNELS}:CUR:0", verbose=verbose, output=False)
        _xpow.send_command(f"CH:1-{XPOW.N_CHANNELS}:VOLT:0", verbose=verbose, output=False)
        if verbose:
            print(f"✅ All {XPOW.N_CHANNELS} XPOW channels turned off.")

# Global XPOW controller instance (singleton)
_xpow = XPOW()
xpow = _xpow


class PhaseShifter:
    """
    Represents a single channel on the photonic chip.
    
    Provides an intuitive interface for controlling individual channels.
    PhaseShifter instances are independent and access the XPOW controller directly.
    
    Parameters
    ----------
    channel_number : int
        Absolute channel number (1-40) on the XPOW controller.
        
    Attributes
    ----------
    channel : int
        The absolute channel number.
    xpow : XPOW
        Reference to the singleton XPOW controller.
        
    Examples
    --------
    >>> ch17 = PhaseShifter(17)
    >>> ch17.set_voltage(2.5)
    >>> current = ch17.get_current()
    """
    
    xpow = _xpow

    def __init__(self, channel_number: int):
        """
        Initialize a PhaseShifter instance.
        
        Parameters
        ----------
        channel_number : int
            Absolute channel number (1-40).
        """
        if not (1 <= channel_number <= XPOW.N_CHANNELS):
            raise ValueError(f"❌ Invalid channel number {channel_number}. Must be between 1 and {XPOW.N_CHANNELS}.")
        self.channel = channel_number
        
    def set_current(self, current: float, verbose: bool = False):
        """
        Set current for this channel.
        
        Parameters
        ----------
        current : float
            Target current in mA.
        verbose : bool, optional
            If True, print command details. Default is False.
        
        Notes
        -----
        The DAC value is computed as: current * CUR_CONVERSION * CUR_CORRECTION[channel]
        where CUR_CONVERSION is a fixed hardware constant and CUR_CORRECTION is calibrable.
        """
        current = max(0, min(_xpow.MAX_CURRENT, current))
        current_value = current * _xpow.CUR_CONVERSION * _xpow.CUR_CORRECTION[self.channel - 1]
        _xpow.send_command(f"CH:{self.channel}:CUR:{int(current_value)}", verbose=verbose, output=False)
        
    def set_voltage(self, voltage: float, verbose: bool = False):
        """
        Set voltage for this channel.
        
        Parameters
        ----------
        voltage : float
            Target voltage in V.
        verbose : bool, optional
            If True, print command details. Default is False.
        
        Notes
        -----
        The DAC value is computed as: voltage * VOLT_CONVERSION * VOLT_CORRECTION[channel]
        where VOLT_CONVERSION is a fixed hardware constant and VOLT_CORRECTION is calibrable.
        """ 
        voltage = max(0, min(_xpow.MAX_VOLTAGE, voltage))
        voltage_value = voltage * _xpow.VOLT_CONVERSION * _xpow.VOLT_CORRECTION[self.channel - 1]
        _xpow.send_command(f"CH:{self.channel}:VOLT:{int(voltage_value)}", verbose=verbose, output=False)
        
    def get_current(self, verbose: bool = False) -> float:
        """
        Query measured current for this channel.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print query details. Default is False.
            
        Returns
        -------
        float
            Measured current in mA.
        """
        res = _xpow.send_command(f"CH:{self.channel}:VAL?", verbose=verbose)
        match = re.search(r'=\s*([\d\.]+)V,\s*([\d\.]+)mA', res)
        if match:
            return float(match.group(2))
        else:
            raise ValueError(f"❌ Unable to parse current from response: {res}")
        
    def get_voltage(self, verbose: bool = False) -> float:
        """
        Query measured voltage for this channel.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print query details. Default is False.
            
        Returns
        -------
        float
            Measured voltage in V.
        """
        res = _xpow.send_command(f"CH:{self.channel}:VAL?", verbose=verbose)
        match = re.search(r'=\s*([\d\.]+)V,\s*([\d\.]+)mA', res)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"❌ Unable to parse voltage from response: {res}")
    
    def turn_off(self, verbose: bool = False):
        """
        Set voltage and current to zero on this channel.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print shutdown commands. Default is False.
            
        Examples
        --------
        >>> ch = Channel(17)
        >>> ch.turn_off()  # Turn off channel 17
        """
        self.set_current(0, verbose=verbose)
        self.set_voltage(0, verbose=verbose)
        if verbose:
            print(f"✅ Channel {self.channel} turned off.")
        
    def ensure_current(self, current: float, tolerance: float = 0.1, max_attempts: int = 100, verbose: bool = False):
        """
        Iteratively adjust current until target is reached.
        
        Parameters
        ----------
        current : float
            Target current in mA.
        tolerance : float, optional
            Acceptable error in mA. Default is 0.1 mA.
        max_attempts : int, optional
            Maximum adjustment iterations. Default is 100.
        verbose : bool, optional
            If True, print adjustment details. Default is False.
            
        Returns
        -------
        float
            Correction factor applied.
        """
        attempts = 0
        step_current = current
        while attempts < max_attempts:
            measured_current = self.get_current(verbose=verbose)
            error = current - measured_current
            if abs(error) <= tolerance:
                return step_current / current
            step = 0.5 * error
            step_current = measured_current + step
            self.set_current(step_current, verbose=verbose)
            attempts += 1
        if abs(error) > tolerance:
            raise RuntimeError(f"❌ Unable to reach target current {current} mA on channel {self.channel} within {tolerance} mA after {max_attempts} attempts.")
        
    def ensure_voltage(self, voltage: float, tolerance: float = 0.01, max_attempts: int = 100, verbose: bool = False):
        """
        Iteratively adjust voltage until target is reached.
        
        Parameters
        ----------
        voltage : float
            Target voltage in V.
        tolerance : float, optional
            Acceptable error in V. Default is 0.01 V.
        max_attempts : int, optional
            Maximum adjustment iterations. Default is 100.
        verbose : bool, optional
            If True, print adjustment details. Default is False.
            
        Returns
        -------
        float
            Correction factor applied.
        """
        attempts = 0
        step_voltage = voltage
        while attempts < max_attempts:
            measured_voltage = self.get_voltage(verbose=verbose)
            error = voltage - measured_voltage
            if abs(error) <= tolerance:
                return step_voltage / voltage
            step = 0.5 * error
            step_voltage = measured_voltage + step
            self.set_voltage(step_voltage, verbose=verbose)
            attempts += 1
        if abs(error) > tolerance:
            raise RuntimeError(f"❌ Unable to reach target voltage {voltage} V on channel {self.channel} within {tolerance} V after {max_attempts} attempts.")
    
    def set_phase(self, phase: float, current: float = 300.0, verbose: bool = False):
        """
        Set phase shift for this channel by varying voltage.
        
        The phase is assumed to be a linear function of the injected power.
        The current is fixed at a specified value (default 300mA), and the
        voltage is adjusted to achieve the desired phase shift.
        
        Parameters
        ----------
        phase : float
            Target phase shift in radians.
        current : float, optional
            Fixed current in mA. Default is 300.0 mA.
        verbose : bool, optional
            If True, print command details. Default is False.
            
        Notes
        -----
        The voltage is computed as: phase * PHASE_CONVERSION[channel]
        where PHASE_CONVERSION is the phase-to-voltage coefficient in V/rad.
        This coefficient can be calibrated using update_phase_coeff().
        """
        # Set the fixed current
        self.set_current(current, verbose=verbose)
        
        # Compute voltage needed for the desired phase
        voltage = phase * _xpow.PHASE_CONVERSION[self.channel - 1]
        
        # Apply the voltage
        self.set_voltage(voltage, verbose=verbose)
        
        if verbose:
            print(f"🔧 Channel {self.channel}: phase={phase:.3f} rad → voltage={voltage:.3f} V @ {current:.1f} mA")
    
    def get_phase(self, current: float = 300.0, verbose: bool = False) -> float:
        """
        Query the current phase shift based on measured voltage.
        
        Parameters
        ----------
        current : float, optional
            Reference current in mA (not actively enforced). Default is 300.0 mA.
        verbose : bool, optional
            If True, print query details. Default is False.
            
        Returns
        -------
        float
            Estimated phase shift in radians, computed from measured voltage.
            
        Notes
        -----
        The phase is computed as: voltage / PHASE_CONVERSION[channel]
        This assumes the voltage-to-phase relationship is linear.
        """
        voltage = self.get_voltage(verbose=verbose)
        phase = voltage / _xpow.PHASE_CONVERSION[self.channel - 1]
        
        if verbose:
            print(f"📊 Channel {self.channel}: voltage={voltage:.3f} V → phase={phase:.3f} rad")
        
        return phase
    
    def update_coeff(self, plot: bool = False, verbose: bool = False):
        """
        Calibrate correction coefficients for this specific channel.
        
        Scans current and voltage, measures actual output, and refines
        CUR_CORRECTION and VOLT_CORRECTION for this channel only.
        
        Parameters
        ----------
        plot : bool, optional
            If True, display fit results. Default is False.
        verbose : bool, optional
            If True, print calibration details. Default is False.
            
        Notes
        -----
        This method only updates the correction coefficient for this channel,
        leaving all other channels unchanged.
        """
        test_currents = np.linspace(1, XPOW.MAX_CURRENT, 10)
        test_voltages = np.linspace(0.1, XPOW.MAX_VOLTAGE, 10)
        
        # Turn off all channels first
        XPOW.turn_off()
        
        # Current calibration
        measured = []
        for c in test_currents:
            self.set_current(c, verbose=verbose)
            val = self.get_current(verbose=verbose)
            measured.append(val)
        measured = np.array(measured)
        coeffs = np.polyfit(test_currents, measured, 1)
        
        old_cur = _xpow.CUR_CORRECTION[self.channel - 1]
        new_cur = old_cur / coeffs[0] if coeffs[0] != 0 else old_cur
        _xpow.CUR_CORRECTION[self.channel - 1] = new_cur
        
        # Voltage calibration
        measured_v = []
        for v in test_voltages:
            self.set_voltage(v, verbose=verbose)
            val = self.get_voltage(verbose=verbose)
            measured_v.append(val)
        measured_v = np.array(measured_v)
        coeffs_v = np.polyfit(test_voltages, measured_v, 1)
        
        old_volt = _xpow.VOLT_CORRECTION[self.channel - 1]
        new_volt = old_volt / coeffs_v[0] if coeffs_v[0] != 0 else old_volt
        _xpow.VOLT_CORRECTION[self.channel - 1] = new_volt
        
        if plot:
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.plot(test_currents, measured, 'o-', label=f'CH{self.channel} Current')
            plt.plot(test_currents, coeffs[0]*test_currents+coeffs[1], '--', label='Fit')
            plt.xlabel('Set Current (mA)')
            plt.ylabel('Measured Current (mA)')
            plt.title(f'Current Calibration CH{self.channel}')
            plt.legend()
            plt.subplot(1,2,2)
            plt.plot(test_voltages, measured_v, 'o-', label=f'CH{self.channel} Voltage')
            plt.plot(test_voltages, coeffs_v[0]*test_voltages+coeffs_v[1], '--', label='Fit')
            plt.xlabel('Set Voltage (V)')
            plt.ylabel('Measured Voltage (V)')
            plt.title(f'Voltage Calibration CH{self.channel}')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        if verbose:
            print(f"✅ Channel {self.channel} calibrated: CUR={new_cur:.4f}, VOLT={new_volt:.4f}")

class Arch:
    """
    Class to handle a specific photonic chip architecture via the XPOW controller.
    
    A Arch instance represents a specific architecture configuration and manages
    a list of PhaseShifter objects corresponding to the TOPAs in that architecture.
    All chip instances share the same XPOW controller connection.
    
    Parameters
    ----------
    arch : int
        Architecture number (1-20). See ARCHS dictionary for available architectures.
        
    Attributes
    ----------
    name : str
        Human-readable name of the architecture.
    id : str
        Architecture identifier code.
    number : int
        Architecture number.
    n_inputs : int
        Number of input ports.
    n_outputs : int
        Number of output ports.
    topas : tuple
        Absolute channel numbers for TOPAs in this architecture.
    channels : list[PhaseShifter]
        List of PhaseShifter instances (indexed from 0).
    xpow : XPOW
        Reference to the singleton XPOW controller.
        
    Examples
    --------
    Create a chip with architecture 6 and control all channels:
    
    >>> chip = Arch(6)
    >>> chip.set_currents([10.0, 15.0, 20.0, 25.0])  # Set all 4 TOPAs
    >>> currents = chip.get_currents()  # Read all TOPA currents
    
    Access individual channels:
    
    >>> chip.channels[0].set_voltage(2.5)  # First TOPA
    >>> chip[1].set_current(50.0)  # Second TOPA (1-indexed via __getitem__)
    
    Notes
    -----
    TOPAs are control using the XPOW controller via a serial connection. 
    The XPOW documentation: https://www.nicslab.com/product-datasheet
    """

    # Data extracted from Nick's lab book
    ARCHS = {
        1:  {
            'name': "Mach-Zehnder Interferometer",
            'id': 'MZI-T12',
            'n_inputs': 1,
            'n_outputs': 1,
            'topas': (1,2),
            },
        2:  {
            'name': "Phase Shifter Solo",
            'id': 'PM-T11',
            'n_inputs': 1,
            'n_outputs': 1,
            'topas': (3,),
            },
        3:  {
            'name': "2x2 MMI Solo",
            'id': 'MMI2x2-T10',
            'n_inputs': 2,
            'n_outputs': 2,
            'topas': (),
            },
        4:  {
            'name': "1x2 MMI Solo",
            'id': 'MMI1x2-T9',
            'n_inputs': 1,
            'n_outputs': 2,
            'topas': (),
            },
        5:  {
            'name': "4-Port Nuller Reconfig",
            'id': 'N4x4-D8',
            'n_inputs': 4,
            'n_outputs': 7,
            'topas': (4,5,6,7,8,9,10,11,12,13,14,15,16),
            },
        6:  {
            'name': "4-Port MMI Active",
            'id': 'N4x4-T8',
            'n_inputs': 4,
            'n_outputs': 4,
            'topas': (17,18,19,20),
            },
        7:  {
            'name': "4-Port Nuller (passive) - FT",
            'id': 'N4x4-D7',
            'n_inputs': 4,
            'n_outputs': 7,
            'topas': (21,22,23,24),
            },
        8:  {
            'name': "Mach-Zender Interferometer",
            'id': 'MZI-T7',
            'n_inputs': 1,
            'n_outputs': 1,
            'topas': (4,5),
            },
        9:  {
            'name': "Normal 4-Port Nuller (active 2x2)",
            'id': 'N2x2-T6',
            'n_inputs': 4,
            'n_outputs': 4,
            'topas': (21,22,23,24,25,26,27,28),
            },
        10: {
            'name': "4-Port Nuller (4x4 MMI) Passive Crazy",
            'id': 'N4x4-D6',
            'n_inputs': 4,
            'n_outputs': 7,
            'topas': (),
            },
        11: {
            'name': "3-Port Kernel Nuller (passive)",
            'id': 'N3x3-D5',
            'n_inputs': 3,
            'n_outputs': 3,
            'topas': (),
            },
        12: {
            'name': "4x4 MMI Passive",
            'id': 'N4x4-T5',
            'n_inputs': 4,
            'n_outputs': 4,
            'topas': (),
            },
        13: {
            'name': "1x2 MMI Passive",
            'id': 'MMI1x2-T4',
            'n_inputs': 1,
            'n_outputs': 2,
            'topas': (),
            },
        14: {
            'name': "Phase Actuator Solo",
            'id': 'PM-T3',
            'n_inputs': 1,
            'n_outputs': 1,
            'topas': (16),
            },
        15: {
            'name': "Mega Kernel Nuller Reconfig",
            'id': 'N2x2-D4',
            'n_inputs': 4,
            'n_outputs': 7,
            'topas': (6,7,33,34,35,36,37,38,28,27,26,25,39,40),
            },
        16: {
            'name': "Kernel Nuller 2x2 Reconfig N",
            'id': 'N2x2-D3',
            'n_inputs': 4,
            'n_outputs': 7,
            'topas': (29, 30, 31, 32),
            },
        17: {
            'name': "Passive Kernel Nuller",
            'id': 'N2x2-D2',
            'n_inputs': 4,
            'n_outputs': 7,
            'topas': (),
            },
        18: {
            'name': "3-Port Kernel Nuller",
            'id': 'N3x3-D1',
            'n_inputs': 3,
            'n_outputs': 3,
            'topas': (),
            },
        19: {
            'name': "2x2 MMI",
            'id': 'N4x4-T2',
            'n_inputs': 4,
            'n_outputs': 4,
            'topas': (),
            },
        20: {
            'name': "2x2 MMI",
            'id': 'MMI2x2-T1',
            'n_inputs': 2,
            'n_outputs': 2,
            'topas': (),
            },
    }

    xpow = _xpow

    def __init__(self, arch: int):
        """
        Initialize a Arch instance for a specific architecture.
        
        Parameters
        ----------
        arch : int
            Architecture number (1-20). See ARCHS dictionary for available architectures.
            
        Raises
        ------
        ValueError
            If the specified architecture number is invalid.
            
        Notes
        -----
        This method automatically creates PhaseShifter objects for all TOPAs in the
        specified architecture and establishes serial connection to the XPOW controller.
        """
        if arch not in Arch.ARCHS:
            raise ValueError(f"❌ Invalid architecture {arch}. Available architectures are: {list(Arch.ARCHS.keys())}")

        self.name = self.ARCHS[arch]['name']
        self.id = self.ARCHS[arch]['id']
        self.number = arch
        self.n_inputs = self.ARCHS[arch]['n_inputs']
        self.n_outputs = self.ARCHS[arch]['n_outputs']
        self.topas = self.ARCHS[arch]['topas']
        
        # Create channel objects (list indexed from 0)
        self.channels = [PhaseShifter(channel_num) for channel_num in self.topas]
        
        # Ensure XPOW connection is established
        _xpow.connect()

    def __getitem__(self, topa_index: int) -> PhaseShifter:
        """
        Access channel by TOPA index (1-indexed): chip[1] returns first TOPA channel.
        
        Parameters
        ----------
        topa_index : int
            TOPA index starting from 1.
            
        Returns
        -------
        Channel
            The corresponding Channel instance.
            
        Raises
        ------
        IndexError
            If topa_index is out of range for this architecture.
        """
        if not (1 <= topa_index <= len(self.channels)):
            raise IndexError(f"❌ TOPA index {topa_index} not available for architecture {self.number}. Available indices: 1-{len(self.channels)}")
        return self.channels[topa_index - 1]
    
    def set_currents(self, currents, verbose: bool = False):
        """
        Set currents for all TOPAs in this chip.
        
        Parameters
        ----------
        currents : array-like
            Array of target currents in mA (one per TOPA).
            Length must match number of TOPAs in architecture.
        verbose : bool, optional
            If True, print command details. Default is False.
            
        Raises
        ------
        ValueError
            If length of currents doesn't match number of TOPAs.
            
        Examples
        --------
        >>> chip = Chip(6)  # 4 TOPAs
        >>> chip.set_currents([10.0, 15.0, 20.0, 25.0])
        """
        currents = np.asarray(currents)
        if len(currents) != len(self.channels):
            raise ValueError(f"❌ Expected {len(self.channels)} current values, got {len(currents)}")
        
        for channel, current in zip(self.channels, currents):
            channel.set_current(current, verbose=verbose)
    
    def set_voltages(self, voltages, verbose: bool = False):
        """
        Set voltages for all TOPAs in this chip.
        
        Parameters
        ----------
        voltages : array-like
            Array of target voltages in V (one per TOPA).
            Length must match number of TOPAs in architecture.
        verbose : bool, optional
            If True, print command details. Default is False.
            
        Raises
        ------
        ValueError
            If length of voltages doesn't match number of TOPAs.
            
        Examples
        --------
        >>> chip = Chip(6)  # 4 TOPAs
        >>> chip.set_voltages([1.5, 2.0, 2.5, 3.0])
        """
        voltages = np.asarray(voltages)
        if len(voltages) != len(self.channels):
            raise ValueError(f"❌ Expected {len(self.channels)} voltage values, got {len(voltages)}")
        
        for channel, voltage in zip(self.channels, voltages):
            channel.set_voltage(voltage, verbose=verbose)
    
    def get_currents(self, verbose: bool = False) -> np.ndarray:
        """
        Query measured currents for all TOPAs in this chip.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print query details. Default is False.
            
        Returns
        -------
        np.ndarray
            Array of measured currents in mA (one per TOPA).
            
        Examples
        --------
        >>> chip = Chip(6)
        >>> currents = chip.get_currents()
        >>> print(currents)  # [10.2, 15.1, 19.8, 24.9]
        """
        return np.array([ch.get_current(verbose=verbose) for ch in self.channels])
    
    def get_voltages(self, verbose: bool = False) -> np.ndarray:
        """
        Query measured voltages for all TOPAs in this chip.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print query details. Default is False.
            
        Returns
        -------
        np.ndarray
            Array of measured voltages in V (one per TOPA).
            
        Examples
        --------
        >>> chip = Chip(6)
        >>> voltages = chip.get_voltages()
        >>> print(voltages)  # [1.52, 2.01, 2.48, 2.99]
        """
        return np.array([ch.get_voltage(verbose=verbose) for ch in self.channels])
    
    def turn_off(self, verbose: bool = False):
        """
        Set voltage and current to zero on all TOPAs in this chip only.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print shutdown commands. Default is False.
            
        Notes
        -----
        This method only affects the channels used by this chip architecture.
        To turn off all 40 XPOW channels, use XPOWController.turn_off().
        
        Examples
        --------
        >>> chip = Chip(6)  # 4 channels: 17, 18, 19, 20
        >>> chip.turn_off()  # Only turns off channels 17-20
        """
        for channel in self.channels:
            channel.set_current(0, verbose=verbose)
            channel.set_voltage(0, verbose=verbose)
        if verbose:
            print(f"✅ Chip {self.name} turned off (channels {list(self.topas)}).")
        if verbose:
            print(f"✅ Chip {self.name} turned off (channels {list(self.topas)}).")

    def set_phases(self, phases, current: float = 300.0, verbose: bool = False):
        """
        Set phase shifts for all TOPAs in this chip.
        
        Parameters
        ----------
        phases : array-like
            Array of target phase shifts in radians (one per TOPA).
            Length must match number of TOPAs in architecture.
        current : float, optional
            Fixed current in mA for all channels. Default is 300.0 mA.
        verbose : bool, optional
            If True, print command details. Default is False.
            
        Raises
        ------
        ValueError
            If length of phases doesn't match number of TOPAs.
            
        Examples
        --------
        >>> chip = Chip(6)  # 4 TOPAs
        >>> chip.set_phases([0.0, np.pi/4, np.pi/2, np.pi])
        """
        phases = np.asarray(phases)
        if len(phases) != len(self.channels):
            raise ValueError(f"❌ Expected {len(self.channels)} phase values, got {len(phases)}")
        
        for channel, phase in zip(self.channels, phases):
            channel.set_phase(phase, current=current, verbose=verbose)
    
    def get_phases(self, current: float = 300.0, verbose: bool = False) -> np.ndarray:
        """
        Query estimated phase shifts for all TOPAs in this chip.
        
        Parameters
        ----------
        current : float, optional
            Reference current in mA (not actively enforced). Default is 300.0 mA.
        verbose : bool, optional
            If True, print query details. Default is False.
            
        Returns
        -------
        np.ndarray
            Array of estimated phase shifts in radians (one per TOPA).
            
        Examples
        --------
        >>> chip = Chip(6)
        >>> phases = chip.get_phases()
        >>> print(phases)  # [0.0, 0.785, 1.571, 3.142]
        """
        return np.array([ch.get_phase(current=current, verbose=verbose) for ch in self.channels])
    
    def update_coeffs(self, plot: bool = False, verbose: bool = False):
        """
        Calibrate correction coefficients for all channels in this chip architecture.
        
        This method only calibrates the channels (TOPAs) used by this specific chip,
        leaving other XPOW channels unchanged.
        
        Parameters
        ----------
        plot : bool, optional
            If True, display fit results for visual inspection. Default is False.
        verbose : bool, optional
            If True, print calibration details. Default is False.
            
        Notes
        -----
        Only updates CORRECTION coefficients for channels in self.topas.
        For calibrating all 40 XPOW channels, use XPOW.update_all_coeffs().
        """
        if verbose:
            print(f"🔧 Calibrating {len(self.channels)} channels for {self.name}...")
        
        for channel in self.channels:
            channel.update_coeff(plot=plot, verbose=verbose)
        
        print(f"✅ Correction coefficients updated for {self.name} (channels {list(self.topas)}).")

# Backward compatibility aliases
class XPOWController(XPOW):
    """
    Deprecated: Use :class:`XPOW` instead.

    .. warning::
        This class is deprecated and will be removed in a future version. Use :class:`XPOW` instead.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("XPOWController is deprecated, use XPOW instead", DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

class Channel(PhaseShifter):
    """
    Deprecated: Use :class:`PhaseShifter` instead.

    .. warning::
        This class is deprecated and will be removed in a future version. Use :class:`PhaseShifter` instead.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("Channel is deprecated, use PhaseShifter instead", DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

class Chip(Arch):
    """
    Deprecated: Use :class:`Arch` instead.

    .. warning::
        This class is deprecated and will be removed in a future version. Use :class:`Arch` instead.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("Chip is deprecated, use Arch instead", DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
