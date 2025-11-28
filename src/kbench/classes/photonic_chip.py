"""
Photonic Chip Control Module.

This module provides a high-level interface for controlling photonic integrated circuits
via the XPOW-8AX-CCvCV-U controller. It supports multiple chip architectures with thermo-optic 
phase actuators (TOPAs) for kernel-nulling interferometry applications.

Classes
-------
Channel
    Represents a single channel on the photonic chip, providing intuitive control of
    voltage and current for individual TOPAs.
    
Chip
    Main class for interfacing with the XPOW controller via serial communication.
    Supports 20 different photonic chip architectures including:
    - Mach-Zehnder Interferometers (MZI)
    - Multi-Mode Interferometers (MMI): 1x2, 2x2, 4x4
    - Kernel Nullers: 2x2, 3x3, 4x4 configurations
    - Phase shifters and phase modulators
    
Key Features
------------
- Channel-wise voltage and current control (0-5V, 0-300mA)
- Architecture-specific channel mapping for TOPAs
- Automatic calibration support for voltage/current coefficients
- Feedback control to ensure setpoints are reached within tolerance
- Sandbox mode for testing without hardware
- Serial communication with XPOW controller (115200 baud)

Hardware
--------
XPOW-8AX-CCvCV-U Controller:
    Product documentation: https://www.nicslab.com/product-datasheet
    - 40 independent voltage/current channels
    - Serial interface (USB, 115200 baud)
    - Voltage range: 0-5V
    - Current range: 0-300mA

Examples
--------
Control a single channel directly:

>>> chip = Chip()
>>> chip[17].set_voltage(2.5)  # Set 2.5V on channel 17
>>> chip[17].ensure_current(50.0, tolerance=0.5)  # Ensure 50mA ±0.5mA

Control multiple channels for architecture 6 (N4x4-T8):

>>> arch = Chip.ARCHS[6]
>>> for topa in arch['topas']:
...     chip[topa].set_current(10.0)

Notes
-----
Voltage and current coefficients are calibrated empirically and stored in CUR_COEFFS
and VOLT_COEFFS arrays. Use update_coeffs() to refine calibration.

See Also
--------
kbench.classes.deformable_mirror : DM control for complementary phase control
kbench.scripts.N4x4_T8.characterize : Characterization script for 4x4 MMI
"""

import numpy as np
from .. import serial
import time
import matplotlib.pyplot as plt
from .. import SANDBOX_MODE
import re

class Channel:
    """
    Represents a single channel on the photonic chip.
    Provides an intuitive interface for controlling individual channels.
    """
    
    def __init__(self, chip_instance, channel_number: int):
        self.chip = chip_instance
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
        """
        Chip.set_current(self.channel, current, verbose=verbose)
        
    def set_voltage(self, voltage: float, verbose: bool = False):
        """
        Set voltage for this channel.
        
        Parameters
        ----------
        voltage : float
            Target voltage in V.
        verbose : bool, optional
            If True, print command details. Default is False.
        """ 
        Chip.set_voltage(self.channel, voltage, verbose=verbose)
        
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
        return Chip.get_current(self.channel, verbose=verbose)
        
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
        return Chip.get_voltage(self.channel, verbose=verbose)
        
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
        return Chip.ensure_current(self.channel, current, tolerance, max_attempts, verbose=verbose)
        
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
        return Chip.ensure_voltage(self.channel, voltage, tolerance, max_attempts, verbose=verbose)

class Chip:
    """
    Class to handle the photonic chip via the XPOW controller.
    TOPAs are control using the XPOW controller via a serial connection. The XPOW documentation can be found in "docs/hardware_documentation/XPOW.pdf".
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

    # These coefficients allow to convert the V and mA values we want to apply into the values we need to send to the XPOW.
    # The following values are "first guess" that has been obtained empricially, but should be refined with a proper calibration using the update_coeffs() function.
    N_CHANNELS = 40
    CUR_COEFFS = np.ones(N_CHANNELS) * 65535 / 210 / 1.418
    VOLT_COEFFS = np.ones(N_CHANNELS) * 65535 / 26 / 1.533

    MAX_VOLTAGE = 5 # V
    MAX_CURRENT = 300 # mA

    SERIAL = None

    def __init__(self, arch:int):
        """
        Initialize a Chip instance for a specific architecture.
        
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
        This method automatically creates Channel objects for all TOPAs in the
        specified architecture and establishes serial connection to the XPOW controller.
        """
        if arch not in Chip.ARCHS:
            raise ValueError(f"❌ Unvalid architecture {arch}. Available architectures are: {list(Chip.ARCHS.keys())}")

        self.name = self.ARCHS[arch]['name']
        self.id = self.ARCHS[arch]['id']
        self.number = arch
        self.n_inputs = self.ARCHS[arch]['n_inputs']
        self.n_outputs = self.ARCHS[arch]['n_outputs']
        self.topas = self.ARCHS[arch]['topas']
        
        # Create channel objects for easy access
        self._channels = {}
        for topa_idx, channel_num in enumerate(self.topas):
            self._channels[topa_idx + 1] = Channel(self, channel_num)

        Chip.connect()

    @classmethod
    def connect(cls):
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
        - Connection is persistent across multiple Chip instances
        """
        if Chip.SERIAL is None:
            # Setup serial connection
            if SANDBOX_MODE:
                port = '/dev/ttyACM0'
            else:
                from serial.tools import list_ports
                port = list(list_ports.grep("2341:8036"))[0][0]   
            Chip.SERIAL = serial.Serial(port, baudrate=115200, timeout=1.0)     
            # Check connexion
            res = cls.send_command("*IDN?")
            if "XPOW" not in res:
                raise ConnectionError(f"❌ No response from the XPOW controller on port {port} at 115200 bauds. Response was: {res}")
        return cls.SERIAL
    
    @classmethod
    def disconnect(cls):
        """
        Close the serial connection to the XPOW controller.
        
        Notes
        -----
        This method is automatically called when the program exits, but can be
        called manually to release the serial port.
        """
        if cls.SERIAL is not None:
            cls.SERIAL.close()
            cls.SERIAL = None
    
    def __getitem__(self, topa_index: int) -> Channel:
        """
        Access channel by TOPA index: c[1] returns first TOPA channel.
        """
        if topa_index not in self._channels:
            raise IndexError(f"❌ TOPA index {topa_index} not available for architecture {self.number}. Available indices: {list(self._channels.keys())}")
        return self._channels[topa_index]

    @classmethod
    def send_command(cls, cmd: str, verbose: bool = False, output=True) -> str:
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
        cls.connect()
        cls.SERIAL.flushInput()
        cls.SERIAL.flushOutput()
        if verbose:
            print(f"📤 XPOW TX: '{cmd}'")
        cls.SERIAL.write(cmd_line.encode())
        time.sleep(0.01)  # Wait a bit for the command to be processed
        if output:
            response = cls.SERIAL.readline().decode().strip()
            if verbose:
                print(f"📥 XPOW RX: '{response}'")
            return response
        else:
            print(f"📥 Output disabled")
            return None


    @classmethod
    def update_coeffs(cls, plot: bool = False, verbose: bool = False):
        """
        Scan current and voltage for all TOPAs, query the XPOW for measured values,
        and refine CUR_COEFFS and VOLT_COEFFS by linear fitting.
        If plot=True, display fit results for visual inspection.
        """

        n = cls.N_CHANNELS
        test_currents = np.linspace(1, cls.MAX_CURRENT, 10)  # mA, avoid 0 for linearity
        test_voltages = np.linspace(0.1, cls.MAX_VOLTAGE, 10)    # V, avoid 0 for linearity

        new_cur_coeffs = np.zeros(n)
        new_volt_coeffs = np.zeros(n)

        for ch in range(1, n+1):
            cls.turn_off()
            # Current calibration
            measured = []
            for c in test_currents:
                cls.set_current(ch, c, verbose=verbose)
                val = cls.get_current(ch, verbose=verbose)
                measured.append(val)
            measured = np.array(measured)
            # Linear fit: measured = a * set + b
            coeffs = np.polyfit(test_currents, measured, 1)
            new_cur_coeffs[ch-1] = cls.CUR_COEFFS[ch-1] / coeffs[0] if coeffs[0] != 0 else cls.CUR_COEFFS[ch-1]

            # Voltage calibration
            measured_v = []
            for v in test_voltages:
                cls.set_voltage(ch, v, verbose=verbose)
                val = cls.get_voltage(ch, verbose=verbose)
                measured_v.append(val)
            measured_v = np.array(measured_v)
            coeffs_v = np.polyfit(test_voltages, measured_v, 1)
            new_volt_coeffs[ch-1] = cls.VOLT_COEFFS[ch-1] / coeffs_v[0] if coeffs_v[0] != 0 else cls.VOLT_COEFFS[ch-1]

            if plot:
                plt.figure(figsize=(10,4))
                plt.subplot(1,2,1)
                plt.plot(test_currents, measured, 'o-', label=f'CH{ch} Current')
                plt.plot(test_currents, coeffs[0]*test_currents+coeffs[1], '--', label='Fit')
                plt.xlabel('Set Current (mA)')
                plt.ylabel('Measured Current (mA)')
                plt.title(f'Current Calibration CH{ch}')
                plt.legend()
                plt.subplot(1,2,2)
                plt.plot(test_voltages, measured_v, 'o-', label=f'CH{ch} Voltage')
                plt.plot(test_voltages, coeffs_v[0]*test_voltages+coeffs_v[1], '--', label='Fit')
                plt.xlabel('Set Voltage (V)')
                plt.ylabel('Measured Voltage (V)')
                plt.title(f'Voltage Calibration CH{ch}')
                plt.legend()
                plt.tight_layout()
                plt.show()

        cls.CUR_COEFFS = new_cur_coeffs
        cls.VOLT_COEFFS = new_volt_coeffs
        print("✅ Coefficients updated.")

    @classmethod
    def set_current(cls, channel: int, current: float, verbose: bool = False, output=False):
        """
        Set current for a specific channel.
        
        Parameters
        ----------
        channel : int
            Absolute channel number (1-40).
        current : float
            Target current in mA. Automatically clamped to [0, MAX_CURRENT].
        verbose : bool, optional
            If True, print command details. Default is False.
        output : bool, optional
            If True, wait for controller response. Default is False.
            
        Notes
        -----
        The actual value sent to the controller is scaled by CUR_COEFFS[channel-1]
        to account for hardware calibration. Values outside [0, 300] mA are clamped.
        """
        current = max(0, min(cls.MAX_CURRENT, current))  # Clamp to valid range
        current_value = current * cls.CUR_COEFFS[channel-1]
        cls.send_command(f"CH:{channel}:CUR:{int(current_value)}", verbose=verbose, output=output)

    @classmethod
    def set_voltage(cls, channel: int, voltage: float, verbose: bool = False, output=False):
        """
        Set voltage for a specific channel.
        
        Parameters
        ----------
        channel : int
            Absolute channel number (1-40).
        voltage : float
            Target voltage in V. Automatically clamped to [0, MAX_VOLTAGE].
        verbose : bool, optional
            If True, print command details. Default is False.
        output : bool, optional
            If True, wait for controller response. Default is False.
            
        Notes
        -----
        The actual value sent to the controller is scaled by VOLT_COEFFS[channel-1]
        to account for hardware calibration. Values outside [0, 5] V are clamped.
        """
        voltage = max(0, min(cls.MAX_VOLTAGE, voltage))  # Clamp to valid range
        voltage_value = voltage * cls.VOLT_COEFFS[channel-1]
        cls.send_command(f"CH:{channel}:VOLT:{int(voltage_value)}", verbose=verbose, output=output)

    @classmethod
    def get_current(cls, channel: int, verbose: bool = False) -> float:
        """
        Query measured current for a specific channel.
        
        Parameters
        ----------
        channel : int
            Absolute channel number (1-40).
        verbose : bool, optional
            If True, print query details. Default is False.
            
        Returns
        -------
        float
            Measured current in mA.
            
        Raises
        ------
        ValueError
            If the controller response cannot be parsed.
            
        Notes
        -----
        Sends "CH:X:VAL?" command and parses response format "=YV, ZmA".
        """
        res = cls.send_command(f"CH:{channel}:VAL?", verbose=verbose)
        # Regex to extract the value from the response
        match = re.search(r'=\s*([\d\.]+)V,\s*([\d\.]+)mA', res)
        if match:
            return float(match.group(2))
        else:
            raise ValueError(f"❌ Unable to parse current from response: {res}")

    @classmethod
    def get_voltage(cls, channel: int, verbose: bool = False) -> float:
        """
        Query measured voltage for a specific channel.
        
        Parameters
        ----------
        channel : int
            Absolute channel number (1-40).
        verbose : bool, optional
            If True, print query details. Default is False.
            
        Returns
        -------
        float
            Measured voltage in V.
            
        Raises
        ------
        ValueError
            If the controller response cannot be parsed.
            
        Notes
        -----
        Sends "CH:X:VAL?" command and parses response format "=YV, ZmA".
        """
        res = cls.send_command(f"CH:{channel}:VAL?", verbose=verbose)
        # Regex to extract the value from the response
        match = re.search(r'=\s*([\d\.]+)V,\s*([\d\.]+)mA', res)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"❌ Unable to parse voltage from response: {res}")

    @classmethod
    def ensure_current(cls, channel: int, current: float, tolerance: float = 0.1, max_attempts: int = 100, verbose: bool = False):
        """
        Iteratively adjust current until target is reached within tolerance.
        
        Parameters
        ----------
        channel : int
            Absolute channel number (1-40).
        current : float
            Target current in mA.
        tolerance : float, optional
            Acceptable error in mA. Default is 0.1 mA.
        max_attempts : int, optional
            Maximum number of adjustment iterations. Default is 100.
        verbose : bool, optional
            If True, print adjustment details. Default is False.
            
        Returns
        -------
        float
            Correction factor applied (final_setpoint / target).
            
        Raises
        ------
        RuntimeError
            If target cannot be reached within tolerance after max_attempts.
            
        Notes
        -----
        Uses proportional control with factor 0.5 to converge to target.
        Useful for compensating thermal drift or controller nonlinearity.
        """
        attempts = 0
        step_current = current
        while attempts < max_attempts:
            measured_current = cls.get_current(channel, verbose=verbose)
            error = current - measured_current
            if abs(error) <= tolerance:
                return step_current / current
            # Simple proportional step, tune factor as needed
            step = 0.5 * error
            step_current = measured_current + step
            cls.set_current(channel, step_current, verbose=verbose)
            attempts += 1
        if abs(error) > tolerance:
            raise RuntimeError(f"❌ Unable to reach target current {current} mA on channel {channel} within {tolerance} mA after {max_attempts} attempts.")
        
    @classmethod
    def ensure_voltage(cls, channel: int, voltage: float, tolerance: float = 0.01, max_attempts: int = 100, verbose: bool = False):
        """
        Iteratively adjust voltage until target is reached within tolerance.
        
        Parameters
        ----------
        channel : int
            Absolute channel number (1-40).
        voltage : float
            Target voltage in V.
        tolerance : float, optional
            Acceptable error in V. Default is 0.01 V.
        max_attempts : int, optional
            Maximum number of adjustment iterations. Default is 100.
        verbose : bool, optional
            If True, print adjustment details. Default is False.
            
        Returns
        -------
        float
            Correction factor applied (final_setpoint / target).
            
        Raises
        ------
        RuntimeError
            If target cannot be reached within tolerance after max_attempts.
            
        Notes
        -----
        Uses proportional control with factor 0.5 to converge to target.
        Useful for compensating thermal drift or controller nonlinearity.
        """
        attempts = 0
        step_voltage = voltage
        while attempts < max_attempts:
            measured_voltage = cls.get_voltage(channel, verbose=verbose)
            error = voltage - measured_voltage
            if abs(error) <= tolerance:
                return step_voltage / voltage
            # Simple proportional step, tune factor as needed
            step = 0.5 * error
            step_voltage = measured_voltage + step
            cls.set_voltage(channel, step_voltage, verbose=verbose)
            attempts += 1
        if abs(error) > tolerance:
            raise RuntimeError(f"❌ Unable to reach target voltage {voltage} V on channel {channel} within {tolerance} V after {max_attempts} attempts.")

    @classmethod
    def turn_off(cls, verbose: bool = False):
        """
        Set voltage and current to zero on all channels.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print shutdown commands. Default is False.
            
        Notes
        -----
        This is a safety method to ensure all TOPAs are powered down.
        Affects all 40 channels simultaneously.
        """
        cls.send_command(f"CH:1-{cls.N_CHANNELS}:CUR:0", verbose=verbose)
        cls.send_command(f"CH:1-{cls.N_CHANNELS}:VOLT:0", verbose=verbose)