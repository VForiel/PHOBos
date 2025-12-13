import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ==========================================
# Configuration & Loading
# ==========================================

# Paths
DATA_PATH = r"/mnt/e/PHOBos/tests/generated/architecture_characterization/4-Port_MMI_Active/20251208_164250/characterization_data.npz"
RESULTS_PATH = "characterization_results.npy"
PLOTS_DIR = "plots"

# Create plots directory if needed
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Load Model Parameters
try:
    model_data = np.load(RESULTS_PATH, allow_pickle=True).item()
    M_matrix = model_data['M']
    C_matrix = model_data['C']
    
    # Load scalar input amplitude or default to 1.0 if missing
    I_ampl = model_data.get('I_ampl', 1.0)
    I_vector = np.ones(4, dtype=complex) * I_ampl
    
    alpha_off = model_data['alpha_off'] # (4,)
    alpha_max = model_data['alpha_max'] # (4,)
    
    print(f"Model loaded successfully. I_ampl={I_ampl:.3f}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load Raw Data
if not os.path.exists(DATA_PATH):
    print(f"Error: Raw data file not found at {DATA_PATH}")
    exit(1)
raw_data = np.load(DATA_PATH)
print("Raw data loaded successfully.")

# ==========================================
# Matrix Model Logic
# ==========================================

def predict_intensities(active_mask, scanned_idx, phases):
    """
    Predict output intensities for a given configuration.
    
    Args:
        active_mask (array-like): Boolean mask of size 4 (True = Active/Max, False = Blocked/Off)
        scanned_idx (int): Index of the input channel being phase-modulated (0-3).
        phases (array-like): Array of phase values (radians) for the scanned input.
        
    Returns:
        np.array: Intensity at the 4 outputs (Shape: [N_phases, 4])
    """
    N = len(phases)
    
    # 1. Input Vector Construction
    # We select specific alpha for each channel based on mask
    mask_arr = np.array(active_mask, dtype=bool) # (4,)
    
    # E_vals[i] = alpha_max[i] if active else alpha_off[i]
    E_vals = np.where(mask_arr, alpha_max, alpha_off) # (4,)
    
    # I_in = I_ampl * E_vals
    I_in_base = I_vector * E_vals
    
    # 2. Phase Modulation
    # The scanned input gets an additional phase term e^{j phi}
    # We create a batch of input vectors (N, 4)
    I_in_batch = np.tile(I_in_base, (N, 1)) # Scan over rows
    
    # Apply phase to the specific column
    # Converting phase to complex rotation: e^{-j phi}
    # Matches the solver convention (which fitted the conjugate term)
    phase_rotations = np.exp(-1j * phases)
    I_in_batch[:, scanned_idx] *= phase_rotations
    
    # 3. Crosstalk Application
    # U = C @ I_in.T
    U_mid = C_matrix @ I_in_batch.T
    
    # 4. Recombination (MMI) Application
    # O = M @ U
    O_out = M_matrix @ U_mid
    
    # 5. Intensity
    intensities = np.abs(O_out.T)**2
    
    return intensities

# ==========================================
# Data Processing & Plotting
# ==========================================

# ==========================================
# Data Processing & Plotting
# ==========================================

# Mapping from Shifter Channel to Input# Shifter Map
# Corrected Mapping (Inverted)
# Raw data uses IDs 17-20
SHIFTER_MAP = {
    17: 3,
    18: 2,
    19: 1,
    20: 0
}
COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e'] # Red, Blue, Green, Orange
LIGHT_COLORS = ['#ff9896', '#aec7e8', '#98df8a', '#ffbb78'] # Lighter versions

def active_indices_1based(mask):
    return [i+1 for i, x in enumerate(mask) if x]

def get_mask_tuple(active_inputs_str):
    """ Parses string '1_2' into tuple (True, True, False, False) """
    active_indices = [int(x)-1 for x in active_inputs_str.split('_')]
    mask = [False]*4
    for idx in active_indices:
        mask[idx] = True
    return tuple(mask)

def generate_summary_plot():
    """ Generates 4 grid plots, grouped by number of active inputs """
    
    # 1. Organize keys by Mask and Shifter
    # Data structure: data_map[mask_tuple][shifter_idx] = key
    data_map = {}
    
    keys = list(raw_data.keys())
    flux_keys = [k for k in keys if k.endswith('_fluxes')]
    
    all_masks = set()
    
    for key in flux_keys:
        match = re.search(r"inputs_([\d_]+)_shifter(\d+)_fluxes", key)
        if not match: continue
        
        active_inputs_str = match.group(1)
        shifter_channel = int(match.group(2))
        
        if shifter_channel not in SHIFTER_MAP: continue
        scanned_idx = SHIFTER_MAP[shifter_channel]
        
        mask = get_mask_tuple(active_inputs_str)
        all_masks.add(mask)
        
        if mask not in data_map:
            data_map[mask] = {}
        data_map[mask][scanned_idx] = key

    # 2. Iterate over groups of active inputs
    # Groups: 1 active, 2 active, 3 active, 4 active
    for num_active in range(1, 5):
        # Filter masks
        group_masks = sorted([m for m in all_masks if sum(m) == num_active], key=lambda m: m)
        
        if not group_masks:
            print(f"No data for {num_active} active inputs.")
            continue
            
        n_cols = len(group_masks)
        n_rows = 4 # Shifters 1-4
        
        print(f"Generating plot for {num_active} Active Inputs. Grid: {n_rows}x{n_cols}")
        
        # 3. Create Figure
        # Width scales with columns
        fig_width = max(5, n_cols * 3)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, n_rows*2.5), constrained_layout=True, squeeze=False)
        fig.suptitle(f"{num_active} Active Inputs", fontsize=16)
        
        # Iterate Row-wise (Shifter) then Col-wise (Mask)
        for r in range(n_rows): # Shifter 0-3
            shifter_idx = r
            
            for c, mask in enumerate(group_masks):
                ax = axs[r, c]
                
                # Setup Axis info
                active_ids = active_indices_1based(mask)
                if r == 0:
                    ax.set_title(f"Inputs: {active_ids}", fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"Scan Input {r+1}", fontsize=10, rotation=90)
                
                # Turn ticks back on
                # ax.set_xticks([]) 
                # ax.set_yticks([])
                
                # Custom X ticks for Phase (0, pi, 2pi)
                ax.set_xticks([0, np.pi, 2*np.pi])
                ax.set_xticklabels(['0', '$\pi$', '$2\pi$'], fontsize=8)
                ax.tick_params(axis='y', labelsize=8)
                
                # Get Data Key
                if shifter_idx in data_map.get(mask, {}):
                    key = data_map[mask][shifter_idx]
                    
                    # --- Plotting Logic ---
                    measured_intensities = raw_data[key]
                    n_points = measured_intensities.shape[0]
                    phases_meas = np.linspace(0, 2*np.pi, n_points)
                    
                    # Model Smooth
                    phases_smooth = np.linspace(0, 2*np.pi, 100)
                    model_smooth = predict_intensities(mask, shifter_idx, phases_smooth)
                    
                    # Model Points
                    model_points = predict_intensities(mask, shifter_idx, phases_meas)
                    
                    # Plot all 4 outputs
                    for out_idx in range(4):
                        # Line (Model)
                        ax.plot(phases_smooth, model_smooth[:, out_idx], color=LIGHT_COLORS[out_idx], linewidth=1.5, alpha=0.8)
                        
                        # Points (Lab) - Dots
                        ax.scatter(phases_meas, measured_intensities[:, out_idx], color=COLORS[out_idx], s=10, marker='o', alpha=0.9)
                        
                        # Points (Model) - Crosses
                        ax.scatter(phases_meas, model_points[:, out_idx], color=LIGHT_COLORS[out_idx], s=15, marker='x', alpha=0.7)
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes, color='gray')

        # Global Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='k', lw=2, label='Model Fit'),
            Line2D([0], [0], marker='x', color='k', label='Model Pts', markersize=5, linestyle='None'),
            Line2D([0], [0], marker='o', color='k', label='Lab Data', markersize=5, linestyle='None'),
        ]
        for i, c in enumerate(COLORS):
           legend_elements.append(Line2D([0], [0], color=c, lw=2, label=f'Out {i+1}'))
           
        fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), ncol=7)
        
        save_path = os.path.join(PLOTS_DIR, f"validation_{num_active}_active_inputs.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Generated: {save_path}")

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    print("Generating validation summary grid...")
    generate_summary_plot()
