import numpy as np
import serial
import time
import matplotlib.pyplot as plt

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

    SERIAL = None

    def __init__(self, arch:int):

        if arch not in Chip.ARCHS:
            raise ValueError(f"❌ Unvalid architecture {arch}. Available architectures are: {list(Chip.ARCHS.keys())}")

        if Chip.SERIAL is None:
            # Setup serial connection
            ...
            # Check connexion
            res = self.send_command("*IDN?")
            if "XPOW" not in res:
                raise ConnectionError(f"❌ No response from the XPOW controller on port {self.port} at {self.baudrate} bauds. Response was: {res}")

        self.name = self.ARCHS[arch]['name']
        self.id = self.ARCHS[arch]['id']
        self.number = arch
        self.n_inputs = self.ARCHS[arch]['n_inputs']
        self.n_outputs = self.ARCHS[arch]['n_outputs']
        self.topas = self.ARCHS[arch]['topas']
        ...

    @staticmethod
    def send_command(cmd:str) -> str:
        # Send a command to the XPOW and return the answer
        cmd += "\n"
        time.sleep(0.01)
        ...

    @staticmethod
    def update_coeffs(plot:bool=False):
        """
        Scan current and voltage for all TOPAs, query the XPOW for measured values,
        and refine CUR_COEFFS and VOLT_COEFFS by linear fitting.
        If plot=True, display fit results for visual inspection.
        """

        n = Chip.N_CHANNELS
        test_currents = np.linspace(10, 300, 6)  # mA, avoid 0 for linearity
        test_voltages = np.linspace(1, 10, 6)    # V, avoid 0 for linearity

        new_cur_coeffs = np.zeros(n)
        new_volt_coeffs = np.zeros(n)

        for ch in range(1, n+1):
            # Current calibration
            measured = []
            for c in test_currents:
                Chip.send_command(f"CH:{ch}:CUR:{int(c * Chip.CUR_COEFFS[ch-1])}")
                time.sleep(0.05)
                res = Chip.send_command(f"CH:{ch}:VAL?")
                try:
                    val = float(res.split(",")[0])  # Assume first value is current in mA
                except Exception:
                    val = np.nan
                measured.append(val)
            measured = np.array(measured)
            # Linear fit: measured = a * set + b
            coeffs = np.polyfit(test_currents, measured, 1)
            new_cur_coeffs[ch-1] = Chip.CUR_COEFFS[ch-1] / coeffs[0] if coeffs[0] != 0 else Chip.CUR_COEFFS[ch-1]

            # Voltage calibration
            measured_v = []
            for v in test_voltages:
                Chip.send_command(f"CH:{ch}:VOLT:{int(v * Chip.VOLT_COEFFS[ch-1])}")
                time.sleep(0.05)
                res = Chip.send_command(f"CH:{ch}:VAL?")
                try:
                    val = float(res.split(",")[1])  # Assume second value is voltage in V
                except Exception:
                    val = np.nan
                measured_v.append(val)
            measured_v = np.array(measured_v)
            coeffs_v = np.polyfit(test_voltages, measured_v, 1)
            new_volt_coeffs[ch-1] = Chip.VOLT_COEFFS[ch-1] / coeffs_v[0] if coeffs_v[0] != 0 else Chip.VOLT_COEFFS[ch-1]

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

        Chip.CUR_COEFFS = new_cur_coeffs
        Chip.VOLT_COEFFS = new_volt_coeffs
        print("✅ Coefficients updated.")

    def set_current(self, topa:int, current:float, abs_channel:bool=False):
        # From 0 to 300 mA
        current = np.max(0, np.min(300, current))  # Clamp to valid range
        channel = self.topas[topa] if not abs_channel else topa
        current = current * Chip.CUR_COEFFS[channel-1]
        Chip.send_command(f"CH:{channel}:CUR:{int(current)}")

    def set_voltage(self, topa:int, voltage:float, abs_channel:bool=False):
        # From 0 to 10 V
        voltage = np.max(0, np.min(10, voltage))  # Clamp to valid range
        channel = self.topas[topa] if not abs_channel else topa
        voltage = voltage * Chip.VOLT_COEFFS[channel-1]
        Chip.send_command(f"CH:{channel}:VOLT:{int(voltage)}")

    def get_current(self, topa:int, abs_channel:bool=False) -> float:
        channel = self.topas[topa] if not abs_channel else topa
        res = Chip.send_command(f"CH:{channel}:VAL?")
        # Regex to extract the value from the response
        ...

    def get_voltage(self, topa:int, abs_channel:bool=False) -> float:
        channel = self.topas[topa] if not abs_channel else topa
        res = Chip.send_command(f"CH:{channel}:VAL?")
        # Regex to extract the value from the response
        ...

    def ensure_current(self, topa:int, current:float, tolerance:float=0.1, abs_channel:bool=False, max_attempts:int=100):
        # Ensure that the current setpoint is reached within the specified tolerance
        attempts = 0
        while attempts < max_attempts:
            measured_current = self.get_current(topa, abs_channel=abs_channel)
            error = current - measured_current
            if abs(error) <= tolerance:
                return
            # Simple proportional step, tune factor as needed
            step = 0.5 * error
            current += step
            self.set_current(topa, current, abs_channel=abs_channel)
            attempts += 1
        if abs(error) > tolerance:
            raise RuntimeError(f"❌ Unable to reach target current {current} mA on TOPA {topa} within {tolerance} mA after {max_attempts} attempts.")
        
    def ensure_voltage(self, topa:int, voltage:float, tolerance:float=0.01, abs_channel:bool=False, max_attempts:int=100):
        # Ensure that the voltage setpoint is reached within the specified tolerance
        attempts = 0
        while attempts < max_attempts:
            measured_voltage = self.get_voltage(topa, abs_channel=abs_channel)
            error = voltage - measured_voltage
            if abs(error) <= tolerance:
                return
            # Simple proportional step, tune factor as needed
            step = 0.5 * error
            voltage += step
            self.set_voltage(topa, voltage, abs_channel=abs_channel)
            attempts += 1
        if abs(error) > tolerance:
            raise RuntimeError(f"❌ Unable to reach target voltage {voltage} V on TOPA {topa} within {tolerance} V after {max_attempts} attempts.")

    @staticmethod
    def turn_off():
        # Turn off all the topas
        for topa in range(1, Chip.N_CHANNELS+1):
            Chip.send_command(f"CH:{topa}:CUR:0")
            Chip.send_command(f"CH:{topa}:VOLT:0")