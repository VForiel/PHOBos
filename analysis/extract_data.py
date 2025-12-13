import numpy as np
import os
import re

# Data Path
# Data Path
DATA_PATH = r"/mnt/e/PHOBos/tests/generated/architecture_characterization/4-Port_MMI_Active/20251208_164250/characterization_data.npz"

def extract_harmonics():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    data = np.load(DATA_PATH)
    
    # Structure to hold processed data
    # List of dictionaries, each representing a "scan"
    # {
    #   'active_inputs_mask': [bool, bool, bool, bool], # Based on nX config
    #   'scanned_input_index': int (0-3), # Which input phase is being scanned (mapped from shifter)
    #   'outputs': [4 complex harmonics] # For the fundamental frequency
    #   'dc_components': [4 float values] # DC average
    # }
    processed_scans = []
    
    # Map Shifter Channel (17-20) to logical Input Index (0-3)
    # Corrected Mapping based on validation results (Inversion)
    SHIFTER_MAP = {
        17: 3,
        18: 2,
        19: 1,
        20: 0
    }
    
    # Regex to parse keys
    # Example: n1_inputs_1_shifter17_phases
    # Example: n4_inputs_1_2_3_4_shifter20_phases
    # We look for keys ending in '_fluxes' to get the intensity data (51 steps x 4 outputs)
    
    keys = list(data.keys())
    flux_keys = [k for k in keys if k.endswith('_fluxes')]
    
    for key in flux_keys:
        # Parse metadata from key name
        # nX_inputs_A_B_..._shifterY_fluxes
        match = re.search(r"inputs_([\d_]+)_shifter(\d+)_fluxes", key)
        if not match:
            continue
            
        active_inputs_str = match.group(1) # e.g. "1_2"
        shifter_channel = int(match.group(2))
        
        # Determine active inputs mask (E matrix mostly)
        active_indices = [int(x)-1 for x in active_inputs_str.split('_')] # 0-based
        active_mask = np.zeros(4, dtype=bool)
        active_mask[active_indices] = True
        
        # Determine scanned input
        if shifter_channel not in SHIFTER_MAP:
            print(f"Warning: Unknown shifter channel {shifter_channel} in key {key}")
            continue
        scanned_input_idx = SHIFTER_MAP[shifter_channel]
        
        # Get Data
        # Shape: (51, 4) -> 51 phase steps, 4 output channels
        intensities = data[key] 
        
        # Extract Harmonics via FFT
        # We expect I = A + B cos(phi + psi)
        # FFT will have peak at index 0 (DC) and index 1 (Fundamental) corresponding to 1 cycle?
        # Need to check phase range. User said "0 to 2pi".
        # If 51 points cover 0 to 2pi, then exactly 1 cycle is the fundamental.
        
        # FFT along axis 0
        ft = np.fft.rfft(intensities, axis=0) / len(intensities)
        
        # DC component is real (ft[0])
        dc = ft[0].real
        
        # Fundamental (index 1) - multiply by 2 to get amplitude for cos form
        # But we keep it as complex coefficient c1 where I ~ c0 + c1 e^{ix} + c1* e^{-ix}
        # typically rfft returns the positive freq term coefficients.
        # If I = A + B cos(x+psi) = A + (B/2)e^{i(x+psi)} + (B/2)e^{-i(x+psi)}
        # Then ft[1] corresponds to (B/2)e^{ipsi}.
        fundamental = ft[1] 
        
        processed_scans.append({
            'active_mask': active_mask,
            'scanned_input_idx': scanned_input_idx,
            'dc': dc,
            'fundamental': fundamental,
            'original_key': key
        })
        
    print(f"Processed {len(processed_scans)} scans.")
    return processed_scans

if __name__ == "__main__":
    scans = extract_harmonics()
    # Save processed data for the solver
    np.save("processed_harmonics.npy", scans)
    print("Saved processed_harmonics.npy")
    
    # Print sample
    s = scans[0]
    print("Sample Scan:")
    print(f"Mask: {s['active_mask']}")
    print(f"Scanned Input: {s['scanned_input_idx']}")
    print(f"DC: {s['dc']}")
    print(f"Fundamental: {s['fundamental']}")
    
