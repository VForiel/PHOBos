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
        """Set current for this channel in mA."""
        Chip.set_current(self.channel, current, verbose=verbose)
        
    def set_voltage(self, voltage: float, verbose: bool = False):
        """Set voltage for this channel in V.""" 
        Chip.set_voltage(self.channel, voltage, verbose=verbose)
        
    def get_current(self, verbose: bool = False) -> float:
        """Get current for this channel in mA."""
        return Chip.get_current(self.channel, verbose=verbose)
        
    def get_voltage(self, verbose: bool = False) -> float:
        """Get voltage for this channel in V."""
        return Chip.get_voltage(self.channel, verbose=verbose)
        
    def ensure_current(self, current: float, tolerance: float = 0.1, max_attempts: int = 100, verbose: bool = False):
        """Ensure current reaches target within tolerance."""
        return Chip.ensure_current(self.channel, current, tolerance, max_attempts, verbose=verbose)
        
    def ensure_voltage(self, voltage: float, tolerance: float = 0.01, max_attempts: int = 100, verbose: bool = False):
        """Ensure voltage reaches target within tolerance."""
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

    MAX_VOLTAGE = 10 # V
    MAX_CURRENT = 300 # mA

    SERIAL = None

    def __init__(self, arch:int):

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
                raise ConnectionError(f"❌ No response from the XPOW controller on port {port} at {baudrate} bauds. Response was: {res}")
        return cls.SERIAL
    
    @classmethod
    def disconnect(cls):
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
    def send_command(cls, cmd: str, verbose: bool = False) -> str:
        # Send a command to the XPOW and return the answer
        cmd_line = cmd + "\n"
        cls.connect()
        if verbose:
            print(f"📤 XPOW TX: '{cmd}'")
        _ = cls.SERIAL.readlines() # Clear the buffer
        cls.SERIAL.write(cmd_line.encode())
        time.sleep(0.01)  # Wait a bit for the command to be processed
        response = cls.SERIAL.readline().decode().strip()
        if verbose:
            print(f"📥 XPOW RX: '{response}'")
        return response

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
    def set_current(cls, channel: int, current: float, verbose: bool = False):
        """Set current for a specific channel (absolute channel number)."""
        current = max(0, min(cls.MAX_CURRENT, current))  # Clamp to valid range
        current_value = current * cls.CUR_COEFFS[channel-1]
        cls.send_command(f"CH:{channel}:CUR:{int(current_value)}", verbose=verbose)

    @classmethod
    def set_voltage(cls, channel: int, voltage: float, verbose: bool = False):
        """Set voltage for a specific channel (absolute channel number)."""
        voltage = max(0, min(cls.MAX_VOLTAGE, voltage))  # Clamp to valid range
        voltage_value = voltage * cls.VOLT_COEFFS[channel-1]
        cls.send_command(f"CH:{channel}:VOLT:{int(voltage_value)}", verbose=verbose)

    @classmethod
    def get_current(cls, channel: int, verbose: bool = False) -> float:
        """Get current for a specific channel (absolute channel number)."""
        res = cls.send_command(f"CH:{channel}:VAL?", verbose=verbose)
        # Regex to extract the value from the response
        match = re.search(r'=\s*([\d\.]+)V,\s*([\d\.]+)mA', res)
        if match:
            return float(match.group(2))
        else:
            raise ValueError(f"❌ Unable to parse current from response: {res}")

    @classmethod
    def get_voltage(cls, channel: int, verbose: bool = False) -> float:
        """Get voltage for a specific channel (absolute channel number)."""
        res = cls.send_command(f"CH:{channel}:VAL?", verbose=verbose)
        # Regex to extract the value from the response
        match = re.search(r'=\s*([\d\.]+)V,\s*([\d\.]+)mA', res)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"❌ Unable to parse voltage from response: {res}")

    @classmethod
    def ensure_current(cls, channel: int, current: float, tolerance: float = 0.1, max_attempts: int = 100, verbose: bool = False):
        """Ensure that the current setpoint is reached within the specified tolerance."""
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
        """Ensure that the voltage setpoint is reached within the specified tolerance."""
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
        """Turn off all channels."""
        for channel in range(1, cls.N_CHANNELS+1):
            cls.send_command(f"CH:{channel}:CUR:0", verbose=verbose)
            cls.send_command(f"CH:{channel}:VOLT:0", verbose=verbose)