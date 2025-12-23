from ..photonic_chip import Arch
import numpy as np
import scipy.optimize

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class Arch6(Arch):
    """
    Architecture 6: 4-Port MMI Active (4 shifters, 4 inputs, 4 outputs).
    
    This is a simplified 4x4 MMI architecture with:
    - 4 inputs (controlled via DM)
    - 4 outputs (all potentially used for beam combining)
    - 4 phase shifters (TOPAs: channels 17, 18, 19, 20)
    """
    
    def __init__(self):
        super().__init__(
            name="4-Port MMI Active",
            id="N4x4-T8",
            n_inputs=4,
            n_outputs=4,
            topas=(17,18,19,20),
            number=6
        )

    def null_calibration_gen(
        self,
        cred3_object,
        crop_centers,
        crop_sizes=10,
        beta: float = 0.8,
        verbose: bool = False,
        plot: bool = False,
        figsize: tuple = (10, 10),
        save_as=None,
    ) -> dict:
        """
        Optimize phase shifter offsets to maximize nulling performance (minimize Null-Depth).
        
        Uses a genetic-like gradient descent algorithm adapted for Arch6.
        Metric: Null-Depth = sum(Nulls) / Bright
        
        Parameters
        ----------
        cred3_object : Cred3
            Camera instance.
        crop_centers : array-like
            Output spot centers (expected 4 spots).
        crop_sizes : int
            Crop window size.
        beta : float
            Decay factor for the step size.
        verbose : bool
            If True, print optimization progress.
        plot : bool
            If True, plot the optimization process.
        figsize : tuple
            Figure size for plots.
        save_as : str
            Path to save the plot if plot is True.
            
        Returns
        -------
        dict
            Dictionary with optimization history (depth, shifters).
        """
        
        if beta < 0.5 or beta >= 1:
            raise ValueError("Beta must be in the range [0.5, 1[")
        
        # Initial step size
        ε = 1e-4 # Minimum shift step size in radians
        Δφ = np.pi / 2 # Initial step
        
        # Arch6 has 4 channels, all participate in optimization
        shifter_indices = range(len(self.channels))
        
        # History
        depth_history = []
        shifters_history = []
        
        # Cache current phases to avoid repeated hardware calls
        current_phases = [ch.get_phase() for ch in self.channels]
        
        def get_metric():
            outs = cred3_object.get_outputs(crop_centers=crop_centers, crop_sizes=crop_sizes)
            # Expected outs: [Bright, Null1, Null2, Null3]
            # Verify we have at least 4 outputs? 
            # Trusting user input on crop_centers for now.
            
            b = outs[0]
            nulls_sum = np.sum(outs[1:])
            
            # Metric: Null-Depth = sum(Nulls) / Bright
            metric = nulls_sum / b if b > 0 else 0
            
            return metric
            
        print("🧬 Starting Genetic Calibration for Arch6...")
        
        iteration_count = 0
        
        while Δφ > ε:
            if verbose:
                print(f"--- Iteration {iteration_count} --- Δφ={Δφ:.2e}")
            
            for i in shifter_indices:
                shifter = self.channels[i]
                
                log = ""
                
                # Measure current state
                m_old = get_metric()
                
                # Get current phase from cache or hardware if first time
                current_phase = current_phases[i]
                
                # Positive step
                shifter.set_phase((current_phase + Δφ) % (2 * np.pi))
                m_pos = get_metric()
                
                # Negative step
                shifter.set_phase((current_phase - Δφ) % (2 * np.pi))
                m_neg = get_metric()
                
                # Restore original position for now
                shifter.set_phase(current_phase)
                
                # Record history
                depth_history.append(m_old)
                shifters_history.append(list(current_phases)) # Use cached values
                
                # Decision logic: Minimize Metric
                updated = False
                log += f"Shift {shifter.channel} Metric: {m_neg:.2e} | {m_old:.2e} | {m_pos:.2e} -> "
                
                if m_pos < m_old and m_pos < m_neg:
                    log += " + "
                    new_phase = (current_phase + Δφ) % (2 * np.pi)
                    shifter.set_phase(new_phase)
                    current_phases[i] = new_phase
                    updated = True
                elif m_neg < m_old and m_neg < m_pos:
                    log += " - "
                    new_phase = (current_phase - Δφ) % (2 * np.pi)
                    shifter.set_phase(new_phase)
                    current_phases[i] = new_phase
                    updated = True
                else:
                    log += " = "
                        
                if verbose:
                    print(log)
            
            # Decay step size
            Δφ *= beta
            iteration_count += 1
            
        print(f"✅ Genetic calibration complete in {iteration_count} iterations.")
        
        if plot and plt is not None:
            shifters_hist_arr = np.array(shifters_history)
            
            _, axs = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
            
            axs[0].plot(depth_history)
            axs[0].set_xlabel("Steps")
            axs[0].set_ylabel("Null-Depth (ΣNulls/Bright)")
            axs[0].set_yscale("log")
            axs[0].set_title("Performance of the Nuller")
            
            for i in range(shifters_hist_arr.shape[1]):
                axs[1].plot(shifters_hist_arr[:, i], label=f"Ch {self.channels[i].channel}")
            
            axs[1].set_xlabel("Steps")
            axs[1].set_ylabel("Phase shift (rad)")
            axs[1].set_title("Convergence of phase shifters")
            axs[1].legend(loc='upper right', bbox_to_anchor=(1,1), fontsize='small', ncol=2)
            
            if save_as:
                plt.savefig(save_as, dpi=150, bbox_inches='tight')
            plt.show()
            
        return {
            "depth": np.array(depth_history),
            "shifters": np.array(shifters_history)
        }

    def solve_matrix(self, data_path=None, plot=True, save_as=None, **kwargs):
        """
        Solve for the interaction matrix A, crosstalk C_before, and input vectors I_ON/OFF.
        
        This method characterizes the component using the Total Flux conservation model:
        O_total = E_in^dag . C_before^dag . P^dag . A . P . C_before . E_in
        
        Where A is the effective Hermitian transfer matrix (M^dag . C_after^dag . C_after . M).
        
        Parameters
        ----------
        data_path : str, optional
            Path to an existing .npz characterization file. If None, runs characterization.
        plot : bool
            If True, plot the comparison between measured and predicted data.
        save_as : str
            Path to save the plot.
        **kwargs : dict
            Arguments to pass to characterize() if data_path is None.
            
        Returns
        -------
        dict
            {'A': matrix (4x4), 'C_before': matrix (4x4), 'I_ON': vector (4,), 'I_OFF': vector (4,), 'cost': float}
        """
        from scipy.optimize import least_squares
        from scipy.linalg import expm, svd
        import re
        
        # 1. Acquire Data
        if data_path is None:
            print("🚀 No data path provided. Running characterization...")
            # Default args for characterization if not provided
            if 'dm_object' not in kwargs or 'cred3_object' not in kwargs:
                raise ValueError("dm_object and cred3_object are required for characterization.")
            
            data_path = self.characterize(plot=False, **kwargs)
            print(f"📂 Data saved to: {data_path}")
            
        # 2. Process Data (Extract Harmonic Coefficients)
        print(f"📊 Loading and processing data from {data_path}...")
        try:
            raw_data = np.load(data_path, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {data_path} not found.")
            
        processed_items = []
        
        # Check if we have the new data format (flattened) or old (nested)
        if 'metadata_scan_keys' in raw_data:
            # New format (flattened keys)
            scan_keys = raw_data['metadata_scan_keys']
            
            for key in scan_keys:
                fluxes = raw_data[f"{key}_fluxes"]
                active_inputs = raw_data[f"{key}_active_inputs"]
                shifter_channel = raw_data[f"{key}_shifter_channel"]
                if hasattr(shifter_channel, 'item'): 
                    # If it's a 0-d array, item() gives scalar.
                    # If it's 1-d array, item() fails or we need to handle it.
                    if shifter_channel.ndim == 0:
                        shifter_channel = shifter_channel.item()
                    elif shifter_channel.size == 1:
                        shifter_channel = shifter_channel.flatten()[0]
                
                # Map shifter channel to 0-3 index
                try:
                    shifter_idx = self.topas.index(shifter_channel)
                except ValueError:
                    # Try converting to int (handle type mismatch like int32 vs int)
                    try:
                        shifter_idx = self.topas.index(int(shifter_channel))
                    except (ValueError, TypeError):
                        print(f"Skipping key {key}. shifter_channel={shifter_channel} not found in {self.topas}.")
                        continue 
                    
                # Construct active mask (size 4)
                active_mask = np.zeros(4, dtype=bool)
                for i in active_inputs:
                    active_mask[i-1] = True
                    
                # FFT to extract harmonics
                N = fluxes.shape[0]
                fft_res = np.fft.fft(fluxes, axis=0) / N
                
                # Keep harmonics for EACH output k
                dc_k = np.abs(fft_res[0]) # Shape (4,)
                fundamental_k = fft_res[1] # Shape (4,)
                
                processed_items.append({
                    'dc': dc_k,
                    'fundamental': fundamental_k,
                    'active_mask': active_mask,
                    'scanned_input_idx': shifter_idx,
                    'key': key,
                    'phases': raw_data.get(f"{key}_phases", None), 
                    'fluxes': fluxes # Store raw for plotting
                })
                
        else:
            # Legacy format support (or unknown format)
            # Iterate over keys in the npz file
            for key in raw_data.files:
                if not key.startswith("n"):
                    continue
                    
                # Each key contains a dictionary-like object (if saved with savez w/ kwargs)
                # or it might be flat arrays if saved differently. 
                # Arch.characterize uses: np.savez(..., **all_scans) where all_scans values are dicts? 
                # No, all_scans values are dicts, but np.savez kwargs saves them as 0-d arrays containing the dict.
                
                try:
                    scan_data = raw_data[key].item()
                except ValueError:
                    # Skip keys that are not scalar objects (e.g. metadata or flattened arrays if mixed)
                    continue
                
                fluxes = scan_data['fluxes'] # Shape (P, 4)
                # phases = scan_data['phases'] # Shape (P,)
                active_inputs = scan_data['active_inputs']
                shifter_channel = scan_data['shifter_channel']
                
                # Map shifter channel to 0-3 index
                try:
                    shifter_idx = self.topas.index(shifter_channel)
                except ValueError:
                    continue 
                    
                # Construct active mask (size 4)
                active_mask = np.zeros(4, dtype=bool)
                for i in active_inputs:
                    active_mask[i-1] = True
                    
                # FFT to extract harmonics
                N = fluxes.shape[0]
                fft_res = np.fft.fft(fluxes, axis=0) / N
                
                # Keep harmonics for EACH output k
                dc_k = np.abs(fft_res[0])
                fundamental_k = fft_res[1]
                
                processed_items.append({
                    'dc': dc_k,
                    'fundamental': fundamental_k,
                    'active_mask': active_mask,
                    'scanned_input_idx': shifter_idx,
                    'key': key,
                    'phases': scan_data.get('phases', None),
                    'fluxes': fluxes
                })
            
        N_sub = len(processed_items)
        if N_sub == 0:
            raise ValueError("No valid scan data found in archive.")
            
        print(f"✅ Extracted {N_sub} data points (Total Flux) for optimization.")
        
        # Prepare arrays for optimization
        measure_dc_arr = np.array([d['dc'] for d in processed_items]) # (N_sub,)
        measure_fund_arr = np.array([d['fundamental'] for d in processed_items]) # (N_sub,)
        active_masks_arr = np.array([d['active_mask'] for d in processed_items], dtype=bool) # (N_sub, 4)
        scanned_indices_arr = np.array([d['scanned_input_idx'] for d in processed_items], dtype=int) # (N_sub,)
        
        # 3. Define Model and Residuals
        
        # 3. Define Model and Residuals
        
        # Helpers for Unitary Parametrization
        # A = exp(iH), where H is Hermitian (16 real params for 4x4)
        # H has 4 real diagonal elements + 6 complex off-diagonal (12 params) = 16 params
        
        def pack_params(H_params, C, I_ON, I_OFF):
            # H_params: 16 floats (4 diag, 6 real off, 6 imag off)
            # C: 32 floats (4x4 complex)
            # I_ON: 8 floats
            # I_OFF: 8 floats
            # Total: 16 + 32 + 8 + 8 = 64 parameters
            return np.concatenate([
                H_params,
                C.real.ravel(), C.imag.ravel(),
                I_ON.real.ravel(), I_ON.imag.ravel(),
                I_OFF.real.ravel(), I_OFF.imag.ravel()
            ])

        def unpack_params(x):
            # H (4x4 Hermitian)
            # Diagonals (4 real)
            h_diag = x[0:4]
            # Off-diagonals (6 complex -> 12 real)
            # Upper triangle indices: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            h_off_real = x[4:10]
            h_off_imag = x[10:16]
            
            H = np.zeros((4,4), dtype=complex)
            # Set diagonals
            np.fill_diagonal(H, h_diag)
            
            # Set off-diagonals
            idx = 0
            for i in range(4):
                for j in range(i+1, 4):
                    val = h_off_real[idx] + 1j * h_off_imag[idx]
                    H[i,j] = val
                    H[j,i] = np.conj(val)
                    idx += 1
            
            # A = exp(iH) is strictly Unitary
            A = expm(1j * H)
            
            # C (4x4 Complex) - 32 params
            ptr = 16
            c_real = x[ptr:ptr+16].reshape(4,4); ptr += 16
            c_imag = x[ptr:ptr+16].reshape(4,4); ptr += 16
            C = c_real + 1j * c_imag
            
            # I_ON - 8 params
            ion_real = x[ptr:ptr+4]; ptr += 4
            ion_imag = x[ptr:ptr+4]; ptr += 4
            I_ON = ion_real + 1j * ion_imag
            
            # I_OFF - 8 params
            ioff_real = x[ptr:ptr+4]; ptr += 4
            ioff_imag = x[ptr:ptr+4]; ptr += 4
            I_OFF = ioff_real + 1j * ioff_imag
            
            return A, C, I_ON, I_OFF

        def compute_residuals(x):
            A, C_before, I_ON, I_OFF = unpack_params(x)
            
            # --- Standard Residuals ---
            # 1. Inputs E_in
            Is_ON = I_ON[None, :]
            Is_OFF = I_OFF[None, :]
            E_in_base = np.where(active_masks_arr, Is_ON, Is_OFF) # (N_sub, 4)
            
            # 2. Pre-Shifter States v = C_before . E_in
            v = (C_before @ E_in_base.T).T # (N_sub, 4)
            
            k_indices = scanned_indices_arr # (N_sub,)
            
            # 3. Model: E_out = A @ P(phi) @ v
            # A_ks = A[:, s]
            A_ks = A[:, k_indices] # (4, N_sub)
            
            # v_s: Input component entering the phase shifter
            v_s = v[np.arange(N_sub), k_indices]
            
            # Z_mod = A_{ks} * v_s
            Z_mod = (A_ks * v_s).T 
            
            # E_0 = A @ v (Field at phi=0)
            E_0 = (A @ v.T).T # (N_sub, 4)
            
            # Z_static = E_0 - Z_mod
            Z_static = E_0 - Z_mod 
            
            # Predicted Harmonics
            pred_dc = np.abs(Z_mod)**2 + np.abs(Z_static)**2
            pred_fund = Z_mod * np.conj(Z_static)
            
            # Measurement Residuals
            diff_dc = (pred_dc - measure_dc_arr).ravel() 
            diff_fund = (pred_fund - measure_fund_arr).ravel()
            
            residuals = np.concatenate([diff_dc, diff_fund.real, diff_fund.imag])
            
            # --- Constraints / Penalties ---
            
            # Constraint: Spectral Norm of C <= 1
            # Penalty = w * max(0, sigma_max - 1)
            # We apply it to all singular values > 1
            s = svd(C_before, compute_uv=False)
            penalty_C = np.maximum(0, s - 1.0) * 1e3 # Strong weight
            
            return np.concatenate([residuals, penalty_C])

        # 4. Run Optimization
        print("🧠 Running constrained optimization (Unitary A, |C|<=1)...")
        
        # Initial Guess
        # A = I => H = 0
        H_init = np.zeros(16) # 4 diag + 12 off (real/imag)
        
        C_init = np.eye(4, dtype=complex)
        I_ON_init = np.ones(4, dtype=complex)
        I_OFF_init = np.zeros(4, dtype=complex)
        
        x0 = pack_params(H_init, C_init, I_ON_init, I_OFF_init)
        
        res = least_squares(compute_residuals, x0, method='lm', max_nfev=5000, verbose=1)
        
        A_final, C_before_final, I_ON_final, I_OFF_final = unpack_params(res.x)
        
        print(f"✅ Optimization complete. Cost: {res.cost:.2e}")

        
        # 5. Plotting (Detailed)
        if plot and plt is not None:
            A, C_before, I_ON, I_OFF = A_final, C_before_final, I_ON_final, I_OFF_final
            
            # Organize data for plotting
            plot_groups = {} 
            input_combos_by_n = {}
            
            for i, item in enumerate(processed_items):
                n_inputs = item['active_mask'].sum()
                if n_inputs not in plot_groups:
                    plot_groups[n_inputs] = []
                    input_combos_by_n[n_inputs] = set()
                item_data = item.copy()
                item_data['global_idx'] = i
                item_data['combo_tuple'] = tuple(item['active_mask'])
                plot_groups[n_inputs].append(item_data)
                input_combos_by_n[n_inputs].add(tuple(item['active_mask']))
                
            for n_inputs in sorted(plot_groups.keys()):
                combos = sorted(list(input_combos_by_n[n_inputs]))
                n_cols = len(combos)
                n_rows = 4 
                
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), 
                                       constrained_layout=True, squeeze=False)
                fig.suptitle(f"Fit Result - {n_inputs} Inputs", fontsize=16)
                
                def predict_flux(phase_val, shifter_idx, active_mask):
                    mask = active_mask
                    E_in = np.where(mask, I_ON, I_OFF)
                    v = C_before @ E_in
                    Pv = v.copy()
                    Pv[shifter_idx] *= np.exp(1j * phase_val)
                    E_out = A @ Pv
                    return np.abs(E_out)**2
                
                for item in plot_groups[n_inputs]:
                    shifter_idx = item['scanned_input_idx']
                    combo_idx = combos.index(item['combo_tuple'])
                    ax = axs[shifter_idx, combo_idx]
                    
                    phases = item['phases']
                    fluxes = item['fluxes']
                    if phases is None: phases = np.linspace(0, 2*np.pi, fluxes.shape[0])
                    
                    colors = ['C0', 'C1', 'C2', 'C3']
                    
                    # Plot Measured
                    for out_ch in range(4):
                        ax.scatter(phases, fluxes[:, out_ch], s=10, alpha=0.5, color=colors[out_ch], label=f'Meas {out_ch+1}')
                    
                    # Plot Model
                    phase_dense = np.linspace(0, 2*np.pi, 50)
                    model_fluxes = np.array([predict_flux(p, shifter_idx, item['active_mask']) for p in phase_dense])
                    
                    for out_ch in range(4):
                        ax.plot(phase_dense, model_fluxes[:, out_ch], '-', color=colors[out_ch], label=f'Mod {out_ch+1}')
                    
                    ax.set_xticks([0, np.pi, 2*np.pi])
                    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
                    
                    # Title
                    active_inputs_list = [k+1 for k in range(4) if item['active_mask'][k]]
                    ax.set_title(f"Scan S{shifter_idx+1}, In{active_inputs_list}", fontsize=10)

                    if shifter_idx == n_rows - 1: 
                        ax.set_xlabel("Phase (rad)")
                    
                    if combo_idx == 0: 
                        ax.set_ylabel("Output (ADU)")
                    
                    if shifter_idx == 0 and combo_idx == n_cols - 1:
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                        
                if save_as:
                    base, ext = os.path.splitext(save_as)
                    fig.savefig(f"{base}_N{n_inputs}{ext}", dpi=150, bbox_inches='tight')
                plt.show()

        return {
            'A': A_final,
            'C_before': C_before_final,
            'I_ON': I_ON_final,
            'I_OFF': I_OFF_final,
            'cost': res.cost
        }