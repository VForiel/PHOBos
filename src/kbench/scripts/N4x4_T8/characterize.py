"""
Characterization script for the N4x4-T8 photonic component.

This script characterizes a 4x4 MMI component with phase shifters on each input.
The component (architecture 6 in photonic_chip.py) has:
- 4 inputs (channels 17, 18, 19, 20 as phase shifters)
- 4 outputs configured as double Bracewell nullers
- 1 constructive output
- 3 destructive (null) outputs

The characterization consists of:
1. Injecting light into single inputs while scanning piston
2. Injecting light into pairs of inputs while scanning piston
3. Recording the response on all 4 outputs
4. Saving plots and data
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import time
from datetime import datetime
import json

from kbench import SANDBOX_MODE
from kbench.classes.deformable_mirror import DM
from kbench.classes.photonic_chip import Chip


def acquire(input_amplitudes: np.ndarray, 
            input_phases: np.ndarray,
            crosstalk: float = 0.05) -> np.ndarray:
    """
    Simulate acquisition from a 4x4 MMI double Bracewell configuration.
    
    The 4x4 MMI implements a kernel-nulling architecture with:
    - Output 0: Constructive combination (all beams in phase)
    - Outputs 1-3: Destructive combinations (double Bracewell nulls)
    
    Parameters
    ----------
    input_amplitudes : np.ndarray
        Amplitude at each of the 4 inputs (length 4).
    input_phases : np.ndarray
        Phase at each of the 4 inputs in radians (length 4).
    crosstalk : float, optional
        Crosstalk coefficient (0-1) representing imperfect nulling.
        Default is 0.05 (5% crosstalk).
    
    Returns
    -------
    np.ndarray
        Intensities at the 4 outputs (length 4).
    
    Notes
    -----
    Double Bracewell configuration creates nulls by combining pairs of inputs
    with π phase shifts. The transfer matrix implements:
    - Output 0: (I1 + I2 + I3 + I4) / 4 (constructive)
    - Output 1: (I1 - I2 + I3 - I4) / 4 (null pattern 1)
    - Output 2: (I1 + I2 - I3 - I4) / 4 (null pattern 2)
    - Output 3: (I1 - I2 - I3 + I4) / 4 (null pattern 3)
    """
    # Create complex field amplitudes
    fields = input_amplitudes * np.exp(1j * input_phases)
    
    # Crosstalk matrix: models coupling between input waveguides
    # Each input loses crosstalk fraction to other inputs
    if crosstalk > 0:
        # Create crosstalk matrix where each off-diagonal element is crosstalk/3
        # (distributing lost power equally to 3 other channels)
        crosstalk_matrix = np.eye(4) * (1 - crosstalk) + \
                          (np.ones((4, 4)) - np.eye(4)) * (crosstalk / 3.0)
    else:
        crosstalk_matrix = np.eye(4)
    
    # Apply crosstalk to input fields
    fields_with_crosstalk = crosstalk_matrix @ fields
    
    # Transfer matrix for double Bracewell (kernel nuller)
    # Rows are outputs, columns are inputs
    transfer_matrix = np.array([
        [ 1,  1,  1,  1],  # Output 0: constructive
        [ 1, -1,  1, -1],  # Output 1: first null
        [ 1,  1, -1, -1],  # Output 2: second null
        [ 1, -1, -1,  1],  # Output 3: third null
    ]) / 2.0
    
    # Compute output fields
    output_fields = transfer_matrix @ fields_with_crosstalk
    
    # Convert to intensities
    intensities = np.abs(output_fields) ** 2
    
    return intensities


def characterize_phase_shifters(chip: Optional[Chip],
                                  wavelength: float = 1550.0,
                                  current: float = 0.3,
                                  v_max: float = 5.0,
                                  n_steps: int = 50,
                                  output_dir: str = "generated/",
                                  verbose: bool = False) -> Dict[str, Dict]:
    """
    Characterize the voltage-to-phase relationship of phase shifters.
    
    This function scans voltage from 0 to v_max at constant current and measures
    the induced phase shift. Results are saved and can be used for calibration.
    
    Parameters
    ----------
    chip : Chip, optional
        Photonic chip instance. If None, simulates with 2π phase at 0.6W.
    wavelength : float, optional
        Wavelength in nm for phase measurement. Default is 1550 nm.
    current : float, optional
        Fixed current in A. Default is 0.3 A.
    v_max : float, optional
        Maximum voltage to scan in V. Default is 5.0 V.
    n_steps : int, optional
        Number of voltage steps. Default is 50.
    output_dir : str, optional
        Directory to save calibration file. Default is "generated/".
    verbose : bool, optional
        If True, print detailed information. Default is False.
    
    Returns
    -------
    dict
        Dictionary with calibration data for each shifter channel.
        Format: {channel_id: {"voltages": [...], "phases": [...], "powers": [...]}}
    
    Notes
    -----
    Simulation model: Phase = 2π * Power / 0.6W, where Power = Voltage * Current
    At constant current (0.3A), Phase = 2π * V * 0.3 / 0.6 = π * V
    """
    # Shifter channels for N4x4-T8 (architecture 6): 17, 18, 19, 20
    shifter_channels = [17, 18, 19, 20]
    
    # Voltage scan
    voltages = np.linspace(0, v_max, n_steps)
    
    calibration_data = {}
    
    print(f"\n{'='*70}")
    print("PHASE SHIFTER CHARACTERIZATION")
    print(f"{'='*70}")
    print(f"Wavelength: {wavelength} nm")
    print(f"Current: {current} A (fixed)")
    print(f"Voltage range: 0 to {v_max} V")
    print(f"Steps: {n_steps}")
    print(f"{'='*70}\n")
    
    for ch_idx, channel in enumerate(shifter_channels):
        print(f"Characterizing shifter {ch_idx + 1}/4 (channel {channel})...")
        
        phases = np.zeros(n_steps)
        powers = np.zeros(n_steps)
        
        for i, voltage in enumerate(voltages):
            power = voltage * current  # Power in Watts
            
            if chip is not None and not SANDBOX_MODE:
                # Real hardware - apply voltage and measure phase
                # This would require interferometric measurement
                chip[channel].set_voltage(voltage)
                chip[channel].set_current(current)
                time.sleep(0.05)
                # TODO: Actual phase measurement from interference pattern
                phases[i] = 0  # Placeholder
            else:
                # Simulation: 2π phase shift at 0.6W
                phases[i] = 2 * np.pi * power / 0.6
            
            powers[i] = power
        
        calibration_data[f"channel_{channel}"] = {
            "channel_id": channel,
            "input_index": ch_idx,
            "voltages": voltages.tolist(),
            "phases": phases.tolist(),
            "powers": powers.tolist(),
            "wavelength": wavelength,
            "current": current,
        }
        
        if verbose:
            print(f"  Voltage range: {voltages[0]:.3f} - {voltages[-1]:.3f} V")
            print(f"  Phase range: {phases[0]:.3f} - {phases[-1]:.3f} rad ({phases[-1]/np.pi:.2f}π)")
            print(f"  Power range: {powers[0]:.3f} - {powers[-1]:.3f} W")
    
    # Save calibration file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    calib_file = output_path / "shifter_calibration.json"
    
    with open(calib_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\nCalibration data saved to {calib_file}")
    
    # Plot calibration curves
    if not verbose:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        for ch_idx, channel in enumerate(shifter_channels):
            ax = axes[ch_idx]
            calib = calibration_data[f"channel_{channel}"]
            voltages = np.array(calib["voltages"])
            phases = np.array(calib["phases"])
            
            ax.plot(voltages, phases / np.pi, 'o-', color='blue', linewidth=2, markersize=3)
            ax.set_xlabel("Voltage (V)", fontsize=12)
            ax.set_ylabel("Phase (π rad)", fontsize=12)
            ax.set_title(f"Input {ch_idx + 1} - Channel {channel}", fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add text annotation for slope
            if len(voltages) > 1:
                slope = (phases[-1] - phases[0]) / (voltages[-1] - voltages[0])
                ax.text(0.05, 0.95, f"Slope: {slope/np.pi:.2f}π/V", 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle("Phase Shifter Calibration", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        # Save figure
        calib_plot_file = output_path / "Shifter_Calibration.png"
        plt.savefig(calib_plot_file, dpi=300, bbox_inches="tight")
        print(f"Calibration plot saved to {calib_plot_file}\n")
        
        plt.show()
    else:
        print()
    
    return calibration_data


def voltage_from_phase(target_phase: float,
                       calibration_data: Dict,
                       channel_key: str) -> float:
    """
    Calculate voltage needed to achieve target phase using calibration data.
    
    Uses linear interpolation between calibration points.
    
    Parameters
    ----------
    target_phase : float
        Desired phase shift in radians.
    calibration_data : dict
        Calibration dictionary from characterize_phase_shifters().
    channel_key : str
        Channel key in format "channel_XX" (e.g., "channel_17").
    
    Returns
    -------
    float
        Voltage in V to achieve target phase.
    """
    calib = calibration_data[channel_key]
    phases = np.array(calib["phases"])
    voltages = np.array(calib["voltages"])
    
    # Clip target phase to calibration range
    target_phase = np.clip(target_phase, phases.min(), phases.max())
    
    # Linear interpolation
    voltage = np.interp(target_phase, phases, voltages)
    
    return voltage


def block_inputs(dm: DM, 
                 segments: List[int], 
                 blocked: List[bool]) -> None:
    """
    Block specified inputs by applying maximum tip/tilt to DM segments.
    
    Parameters
    ----------
    dm : DM
        Deformable mirror instance.
    segments : List[int]
        List of 4 segment IDs corresponding to the 4 inputs.
    blocked : List[bool]
        List of 4 booleans indicating which inputs to block.
    """
    max_tiptilt = 10.0  # milliradians - adjust based on hardware
    
    for i, (seg_id, is_blocked) in enumerate(zip(segments, blocked)):
        if is_blocked:
            # Apply large tip/tilt to deflect beam
            dm[seg_id].set_ptt(piston=0, tip=max_tiptilt, tilt=max_tiptilt)
        else:
            # Reset to flat
            dm[seg_id].set_ptt(piston=0, tip=0, tilt=0)


def scan_piston_dm(dm: Optional[DM],
                   segments: List[int],
                   active_inputs: List[int],
                   wavelength: float = 1550.0,
                   n_steps: int = 50,
                   crosstalk: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan piston from 0 to 2λ using DM segments.
    
    Parameters
    ----------
    dm : DM, optional
        Deformable mirror instance. If None and in sandbox mode, simulates scan.
    segments : List[int]
        List of 4 segment IDs for the 4 inputs.
    active_inputs : List[int]
        Indices of inputs to activate (0-3).
    wavelength : float, optional
        Wavelength in nm. Default is 1550 nm.
    n_steps : int, optional
        Number of piston steps. Default is 50.
    crosstalk : float, optional
        Crosstalk coefficient. Default is 0.05.
    
    Returns
    -------
    pistons : np.ndarray
        Array of piston values in nm (length n_steps).
    intensities : np.ndarray
        Array of intensities at 4 outputs (shape: n_steps x 4).
    """
    # Piston scan range: 0 to 2λ
    pistons = np.linspace(0, 2 * wavelength, n_steps)
    intensities = np.zeros((n_steps, 4))
    
    # Input amplitudes (1 for active, 0 for inactive)
    input_amps = np.array([1.0 if i in active_inputs else 0.0 for i in range(4)])
    
    # NOTE: We do NOT block inactive inputs - they stay open for crosstalk
    # This allows interference between the active input and crosstalk signals
    
    for i, piston in enumerate(pistons):
        if dm is not None and not SANDBOX_MODE:
            # Apply piston to highest numbered active input
            dm[segments[active_inputs[-1]]].set_piston(piston)
            # Keep other active inputs at 0 piston
            for inp_idx in active_inputs[:-1]:
                dm[segments[inp_idx]].set_piston(0)
            time.sleep(0.05)  # Stabilization time
        
        # Phase from piston (2π per wavelength)
        # The highest numbered active input gets the piston scan, others stay at 0
        phases = np.zeros(4)
        phases[active_inputs[-1]] = 2 * np.pi * piston / wavelength
        # Other active inputs stay at phase 0 (set by their amplitude)
        
        # Acquire intensities (simulated or real)
        intensities[i] = acquire(input_amps, phases, crosstalk)
    
    # Reset DM if in real mode
    if dm is not None and not SANDBOX_MODE:
        for seg_id in segments:
            dm[seg_id].set_ptt(0, 0, 0)
    
    return pistons, intensities


def scan_piston_shifters(chip: Optional[Chip],
                          active_inputs: List[int],
                          calibration_data: Optional[Dict] = None,
                          wavelength: float = 1550.0,
                          n_steps: int = 50,
                          crosstalk: float = 0.0,
                          current: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan piston from 0 to 2λ using phase shifters with calibration.
    
    Parameters
    ----------
    chip : Chip, optional
        Photonic chip instance. If None and in sandbox mode, simulates scan.
    active_inputs : List[int]
        Indices of inputs to activate (0-3).
    calibration_data : dict, optional
        Calibration data from characterize_phase_shifters(). If None, uses
        simple linear model (phase proportional to voltage).
    wavelength : float, optional
        Wavelength in nm. Default is 1550 nm.
    n_steps : int, optional
        Number of piston steps. Default is 50.
    crosstalk : float, optional
        Crosstalk coefficient. Default is 0.0.
    current : float, optional
        Current in A. Default is 0.3 A.
    
    Returns
    -------
    pistons : np.ndarray
        Array of equivalent piston values in nm (length n_steps).
    intensities : np.ndarray
        Array of intensities at 4 outputs (shape: n_steps x 4).
    """
    # Shifter channels for N4x4-T8 (architecture 6): 17, 18, 19, 20
    shifter_channels = [17, 18, 19, 20]
    
    # Phase scan from 0 to 4π (equivalent to 2λ piston)
    target_phases = np.linspace(0, 4 * np.pi, n_steps)
    pistons = np.linspace(0, 2 * wavelength, n_steps)
    intensities = np.zeros((n_steps, 4))
    
    # Input amplitudes
    input_amps = np.array([1.0 if i in active_inputs else 0.0 for i in range(4)])
    
    # Determine voltages for each phase step
    if calibration_data is not None:
        # Use calibration data with interpolation
        scanned_channel_key = f"channel_{shifter_channels[active_inputs[-1]]}"
        voltages = np.array([voltage_from_phase(phase, calibration_data, scanned_channel_key) 
                            for phase in target_phases])
    else:
        # Simple linear model: phase = π * V (at current = 0.3A)
        voltages = target_phases / np.pi
    
    if chip is not None and not SANDBOX_MODE:
        # Reset all shifters in real mode
        for ch in shifter_channels:
            chip[ch].set_voltage(0.0)
            chip[ch].set_current(current)
    
    for i, (voltage, target_phase) in enumerate(zip(voltages, target_phases)):
        if chip is not None and not SANDBOX_MODE:
            # Apply voltage to highest numbered active shifter
            chip[shifter_channels[active_inputs[-1]]].set_voltage(voltage)
            chip[shifter_channels[active_inputs[-1]]].set_current(current)
            # Keep other active shifters at 0
            for inp_idx in active_inputs[:-1]:
                chip[shifter_channels[inp_idx]].set_voltage(0.0)
                chip[shifter_channels[inp_idx]].set_current(current)
            time.sleep(0.05)  # Stabilization time
        
        # Phase from target (already calculated)
        # The highest numbered active input gets the phase scan, others stay at 0
        phases = np.zeros(4)
        phases[active_inputs[-1]] = target_phase
        
        # Acquire intensities (simulated or real)
        intensities[i] = acquire(input_amps, phases, crosstalk)
    
    # Reset shifters if in real mode
    if chip is not None and not SANDBOX_MODE:
        for ch in shifter_channels:
            chip[ch].set_voltage(0.0)
    
    return pistons, intensities


def characterize_single_inputs(dm: Optional[DM] = None,
                                chip: Optional[Chip] = None,
                                segments: List[int] = [111, 112, 113, 114],
                                use_shifters: bool = False,
                                calibration_data: Optional[Dict] = None,
                                wavelength: float = 1550.0,
                                n_steps: int = 50,
                                crosstalk: float = 0.0,
                                output_dir: str = "generated/") -> dict:
    """
    Characterize response for each single input.
    
    With only one input active, the 4x4 MMI acts as a 1x4 splitter.
    We expect to see constant intensities at all outputs (no interference fringes)
    with ~25% of the input power at each output (in absence of losses).
    
    Parameters
    ----------
    dm : DM, optional
        Deformable mirror instance (required if use_shifters=False).
    chip : Chip, optional
        Photonic chip instance (required if use_shifters=True).
    segments : List[int], optional
        DM segment IDs for the 4 inputs. Default is [111, 112, 113, 114].
    use_shifters : bool, optional
        Use phase shifters instead of DM. Default is False.
    wavelength : float, optional
        Wavelength in nm. Default is 1550 nm.
    n_steps : int, optional
        Number of piston steps. Default is 50.
    crosstalk : float, optional
        Crosstalk coefficient. Default is 0.05.
    output_dir : str, optional
        Directory to save results. Default is "generated/".
    
    Returns
    -------
    dict
        Dictionary with results for each input.
    """
    results = {}
    
    for input_idx in range(4):
        print(f"Characterizing input {input_idx + 1}/4...")
        
        # Only one active input (true single input test)
        active = [input_idx]
        
        if use_shifters:
            if chip is None and not SANDBOX_MODE:
                raise ValueError("Chip instance required when use_shifters=True")
            pistons, intensities = scan_piston_shifters(
                chip, active, calibration_data, wavelength, n_steps, crosstalk
            )
        else:
            if dm is None and not SANDBOX_MODE:
                raise ValueError("DM instance required when use_shifters=False")
            pistons, intensities = scan_piston_dm(
                dm, segments, active, wavelength, n_steps, crosstalk
            )
        
        # Use 1-indexed naming (inputs 1-4 instead of 0-3)
        results[f"input_{input_idx + 1}"] = {
            "pistons": pistons,
            "intensities": intensities,
        }
    
    return results


def characterize_dual_inputs(dm: Optional[DM] = None,
                              chip: Optional[Chip] = None,
                              segments: List[int] = [111, 112, 113, 114],
                              use_shifters: bool = False,
                              calibration_data: Optional[Dict] = None,
                              wavelength: float = 1550.0,
                              n_steps: int = 50,
                              crosstalk: float = 0.0,
                              output_dir: str = "generated/") -> dict:
    """
    Characterize response for all pairs of inputs.
    
    Parameters
    ----------
    dm : DM, optional
        Deformable mirror instance (required if use_shifters=False).
    chip : Chip, optional
        Photonic chip instance (required if use_shifters=True).
    segments : List[int], optional
        DM segment IDs for the 4 inputs. Default is [111, 112, 113, 114].
    use_shifters : bool, optional
        Use phase shifters instead of DM. Default is False.
    wavelength : float, optional
        Wavelength in nm. Default is 1550 nm.
    n_steps : int, optional
        Number of piston steps. Default is 50.
    crosstalk : float, optional
        Crosstalk coefficient. Default is 0.05.
    output_dir : str, optional
        Directory to save results. Default is "generated/".
    
    Returns
    -------
    dict
        Dictionary with results for each input pair.
    """
    results = {}
    
    # Generate all pairs of inputs
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    
    for idx, (i, j) in enumerate(pairs):
        # Display 1-indexed for user
        print(f"Characterizing input pair ({i + 1}, {j + 1}) - {idx + 1}/{len(pairs)}...")
        
        if use_shifters:
            if chip is None and not SANDBOX_MODE:
                raise ValueError("Chip instance required when use_shifters=True")
            pistons, intensities = scan_piston_shifters(
                chip, [i, j], calibration_data, wavelength, n_steps, crosstalk
            )
        else:
            if dm is None and not SANDBOX_MODE:
                raise ValueError("DM instance required when use_shifters=False")
            pistons, intensities = scan_piston_dm(
                dm, segments, [i, j], wavelength, n_steps, crosstalk
            )
        
        # Use 1-indexed naming (inputs 1-4 instead of 0-3)
        results[f"inputs_{i + 1}_{j + 1}"] = {
            "pistons": pistons,
            "intensities": intensities,
        }
    
    return results


def create_output_directory(base_dir: str = "generated/N4x4_T8_characterization/") -> Path:
    """
    Create timestamped output directory with run number.
    
    Format: YYYY.MM.DD-NNN where NNN is incremented for each run on the same day.
    
    Parameters
    ----------
    base_dir : str
        Base directory for all characterization runs.
    
    Returns
    -------
    Path
        Path to the created directory.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Get current date
    now = datetime.now()
    date_str = now.strftime("%Y.%m.%d")
    
    # Find existing directories for today
    existing = list(base_path.glob(f"{date_str}-*"))
    
    if existing:
        # Extract run numbers and find max
        run_numbers = []
        for p in existing:
            try:
                num = int(p.name.split("-")[-1])
                run_numbers.append(num)
            except ValueError:
                continue
        next_run = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_run = 1
    
    # Create new directory
    run_dir = base_path / f"{date_str}-{next_run:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_parameters(output_dir: Path, **kwargs) -> None:
    """
    Save characterization parameters to JSON file.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save parameters file.
    **kwargs
        Parameters to save.
    """
    params_file = output_dir / "parameters.json"
    
    # Add timestamp
    params = {"timestamp": datetime.now().isoformat(), **kwargs}
    
    # Convert numpy arrays and non-serializable types
    serializable_params = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            serializable_params[key] = value.tolist()
        elif isinstance(value, Path):
            serializable_params[key] = str(value)
        else:
            serializable_params[key] = value
    
    with open(params_file, 'w') as f:
        json.dump(serializable_params, f, indent=2)
    
    print(f"Parameters saved to {params_file}")


def plot_results(results: dict,
                 title: str,
                 output_dir: str = "generated/",
                 verbose: bool = False) -> None:
    """
    Plot and save characterization results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from characterization.
    title : str
        Title for the plot.
    output_dir : str, optional
        Directory to save plots. Default is "generated/".
    verbose : bool, optional
        If True, print debug information instead of plotting. Default is False.
    """
    output_labels = ["Constructive", "Null 1", "Null 2", "Null 3"]
    
    if verbose:
        # DEBUG: Print results instead of plotting
        print(f"\n{'='*70}")
        print(f"DEBUG: {title}")
        print(f"{'='*70}")
        
        for config_name, data in results.items():
            print(f"\n{config_name}:")
            pistons = data["pistons"]
            intensities = data["intensities"]
            
            # Show first, middle, and last piston values
            indices = [0, len(pistons)//2, -1]
            for idx in indices:
                print(f"  Piston = {pistons[idx]:.1f} nm:")
                for i in range(4):
                    print(f"    {output_labels[i]:15s}: {intensities[idx, i]:.6f}")
            
            # Show min, max, mean, std for each output
            print(f"  Statistics over full scan:")
            for i in range(4):
                print(f"    {output_labels[i]:15s}: min={intensities[:, i].min():.6f}, max={intensities[:, i].max():.6f}, mean={intensities[:, i].mean():.6f}, std={intensities[:, i].std():.6f}")
        
        print(f"{'='*70}\n")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    n_configs = len(results)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows))
    if n_configs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = ["green", "red", "blue", "orange"]
    markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond
    
    for idx, (config_name, data) in enumerate(results.items()):
        ax = axes[idx]
        pistons = data["pistons"]
        intensities = data["intensities"]
        
        # Extract input indices from config_name (already 1-indexed in the name)
        # Format: "input_X" or "inputs_X_Y"
        if config_name.startswith("input_") and "inputs_" not in config_name:
            # Single input characterization: "input_1"
            parts = config_name.split("_")
            input_num = parts[1]
            xlabel = f"Piston on input {input_num} (nm)"
        elif "inputs_" in config_name:
            # Dual input characterization: "inputs_1_2"
            parts = config_name.split("_")
            input1 = parts[1]
            input2 = parts[2]
            xlabel = f"Piston on input {input2} (nm)"
        else:
            xlabel = "Piston (nm)"
        
        # Plot each output with alpha=0.5 and distinct markers
        for i in range(4):
            ax.plot(pistons, intensities[:, i], 
                   label=output_labels[i], color=colors[i], linewidth=2, alpha=0.5,
                   marker=markers[i], markersize=4, markevery=max(1, len(pistons)//10))
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Intensity (a.u.)", fontsize=12)
        ax.set_title(config_name.replace("_", " ").title(), fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    # Save figure
    filename = f"{title.replace(' ', '_')}.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {filepath}")
    
    plt.show()


def run_full_characterization(dm: Optional[DM] = None,
                               chip: Optional[Chip] = None,
                               segments: List[int] = [111, 112, 113, 114],
                               use_shifters: bool = False,
                               wavelength: float = 1550.0,
                               n_steps: int = 50,
                               crosstalk: float = 0.0,
                               output_dir: str = "generated/",
                               verbose: bool = False) -> None:
    """
    Run full characterization: single inputs and dual inputs.
    
    Parameters
    ----------
    dm : DM, optional
        Deformable mirror instance (required if use_shifters=False).
    chip : Chip, optional
        Photonic chip instance (required if use_shifters=True).
    segments : List[int], optional
        DM segment IDs for the 4 inputs. Default is [111, 112, 113, 114].
    use_shifters : bool, optional
        Use phase shifters instead of DM. Default is False.
    wavelength : float, optional
        Wavelength in nm. Default is 1550 nm.
    n_steps : int, optional
        Number of piston steps. Default is 50.
    crosstalk : float, optional
        Crosstalk coefficient (0-1). Default is 0.0 (0%).
    output_dir : str, optional
        Directory to save results. Default is "generated/".
    verbose : bool, optional
        If True, print debug information instead of plotting. Default is False.
    
    Examples
    --------
    # Using DM for piston scan
    >>> from kbench.classes.deformable_mirror import DM
    >>> dm = DM()
    >>> run_full_characterization(dm=dm, segments=[111, 112, 113, 114])
    
    # Using phase shifters
    >>> from kbench.classes.photonic_chip import Chip
    >>> chip = Chip()
    >>> run_full_characterization(chip=chip, use_shifters=True)
    """
    # Create timestamped output directory
    if output_dir == "generated/":
        output_path = create_output_directory()
    else:
        output_path = create_output_directory(output_dir)
    
    # Save parameters
    save_parameters(
        output_path,
        wavelength=wavelength,
        n_steps=n_steps,
        crosstalk=crosstalk,
        use_shifters=use_shifters,
        segments=segments if not use_shifters else None,
        component="N4x4-T8",
        description="4x4 MMI with phase shifters - Double Bracewell configuration"
    )
    
    print("=" * 70)
    print("N4x4-T8 Component Characterization")
    print("=" * 70)
    print(f"Wavelength: {wavelength} nm")
    print(f"Piston steps: {n_steps}")
    print(f"Crosstalk: {crosstalk * 100:.1f}%")
    print(f"Method: {'Phase shifters' if use_shifters else 'DM segments'}")
    if not use_shifters:
        print(f"DM segments: {segments}")
    print(f"Output directory: {output_path}")
    print("=" * 70)
    
    # Characterize phase shifters if using them
    calibration_data = None
    if use_shifters:
        calibration_data = characterize_phase_shifters(
            chip, 
            wavelength=wavelength, 
            current=0.3,
            v_max=5.0,
            n_steps=50,
            output_dir=str(output_path),
            verbose=verbose
        )
    
    # Characterize single inputs
    print("\n### SINGLE INPUT CHARACTERIZATION ###")
    single_results = characterize_single_inputs(
        dm, chip, segments, use_shifters, calibration_data, wavelength, n_steps, crosstalk, str(output_path)
    )
    plot_results(single_results, "Single Input Characterization", str(output_path), verbose)
    
    # Characterize dual inputs
    print("\n### DUAL INPUT CHARACTERIZATION ###")
    dual_results = characterize_dual_inputs(
        dm, chip, segments, use_shifters, calibration_data, wavelength, n_steps, crosstalk, str(output_path)
    )
    plot_results(dual_results, "Dual Input Characterization", str(output_path), verbose)
    
    print("\n" + "=" * 70)
    print("Characterization complete!")
    print("=" * 70)



if __name__ == "__main__":
    if SANDBOX_MODE:
        print("⛱️  Running in SANDBOX mode (simulation only)")
        print("\nThis script simulates the characterization of the N4x4-T8 component.")
        print("To use with real hardware, install BMC SDK and instantiate DM or Chip objects.\n")
        
        # Simulate with mock data
        run_full_characterization(
            dm=None,
            chip=None,
            segments=[111, 112, 113, 114],
            use_shifters=False,
            wavelength=1550.0,
            n_steps=30,
            crosstalk=0.5,
            output_dir="generated/N4x4_T8_characterization/"
        )
    else:
        print("✅ Running in CONTROL mode (hardware available)")
        print("\nInitializing hardware for N4x4-T8 characterization...\n")
        
        # Real hardware mode - create instances
        try:
            dm = DM()
            run_full_characterization(
                dm=dm,
                chip=None,
                segments=[111, 112, 113, 114],
                use_shifters=False,
                wavelength=1550.0,
                n_steps=50,
                crosstalk=0.0,
                output_dir="generated/N4x4_T8_characterization/"
            )
        except Exception as e:
            print(f"Error during hardware operation: {e}")
            print("\nFalling back to simulation mode...\n")
            run_full_characterization(
                dm=None,
                chip=None,
                segments=[111, 112, 113, 114],
                use_shifters=False,
                wavelength=1550.0,
                n_steps=30,
                crosstalk=0.0,
                output_dir="generated/N4x4_T8_characterization/"
            )
