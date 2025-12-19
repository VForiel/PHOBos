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
        dm_object,
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
        dm_object : DM
            Deformable mirror instance.
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
        Solve for the interaction matrix A and input vector I using characterization data.
        
        This method characterizes the component (or loads existing data), processes the
        raw flux scans into harmonics (DC and Fundamental), and solves the system
        O = A . T . I using least squares optimization.
        
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
            {'A': matrix (4x4), 'I_ON': vector (4,), 'I_OFF': vector (4,)}
        """
        from scipy.optimize import least_squares
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
        
        # Iterate over keys in the npz file
        for key in raw_data.files:
            if not key.startswith("n"):
                continue
                
            # Each key contains a dictionary-like object (if saved with savez w/ kwargs)
            # or it might be flat arrays if saved differently. 
            # Arch.characterize uses: np.savez(..., **all_scans) where all_scans values are dicts? 
            # No, all_scans values are dicts, but np.savez kwargs saves them as 0-d arrays containing the dict.
            
            scan_data = raw_data[key].item()
            
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
            
            dc = np.abs(fft_res[0]) # Real
            fundamental = fft_res[1] # Complex (corresponding to e^-iphi modulation)
            
            processed_items.append({
                'dc': dc,
                'fundamental': fundamental,
                'active_mask': active_mask,
                'scanned_input_idx': shifter_idx,
                'key': key
            })
            
        N_sub = len(processed_items)
        if N_sub == 0:
            raise ValueError("No valid scan data found in archive.")
            
        print(f"✅ Extracted {N_sub} data points for optimization.")
        
        # Prepare arrays for optimization
        measure_dc_arr = np.array([d['dc'] for d in processed_items]) # (N_sub, 4)
        measure_fund_arr = np.array([d['fundamental'] for d in processed_items]) # (N_sub, 4)
        active_masks_arr = np.array([d['active_mask'] for d in processed_items], dtype=bool) # (N_sub, 4)
        scanned_indices_arr = np.array([d['scanned_input_idx'] for d in processed_items], dtype=int) # (N_sub,)
        
        # 3. Define Model and Residuals
        
        def compute_residuals(A, I_ON, I_OFF):
            Is_ON = I_ON[None, :]
            Is_OFF = I_OFF[None, :]
            I_base = np.where(active_masks_arr, Is_ON, Is_OFF) # (N_sub, 4)
            
            O_base = (A @ I_base.T).T # (N_sub, 4)
            
            k_indices = scanned_indices_arr
            I_k = I_base[np.arange(N_sub), k_indices] # (N_sub,)
            A_cols = A[:, k_indices].T # (N_sub, 4)
            
            O_mod_phys = A_cols * I_k[:, None] # (N_sub, 4)
            O_static_phys = O_base - O_mod_phys # (N_sub, 4)
            
            pred_dc = np.abs(O_static_phys)**2 + np.abs(O_mod_phys)**2
            pred_fund = O_static_phys * np.conj(O_mod_phys)
            
            diff_dc = (pred_dc - measure_dc_arr).ravel()
            diff_fund = (pred_fund - measure_fund_arr).ravel()
            
            return np.concatenate([diff_dc, diff_fund.real, diff_fund.imag])

        # Helpers
        def pack(A, I_ON, I_OFF):
            return np.hstack([
                A.real.ravel(), A.imag.ravel(),
                I_ON.real.ravel(), I_ON.imag.ravel(),
                I_OFF.real.ravel(), I_OFF.imag.ravel()
            ])
            
        def unpack(x):
            idx = 0
            A_real = x[idx:idx+16].reshape(4,4); idx += 16
            A_imag = x[idx:idx+16].reshape(4,4); idx += 16
            A = A_real + 1j * A_imag
            
            ion_real = x[idx:idx+4]; idx += 4
            ion_imag = x[idx:idx+4]; idx += 4
            I_ON = ion_real + 1j * ion_imag
            
            ioff_real = x[idx:idx+4]; idx += 4
            ioff_imag = x[idx:idx+4]; idx += 4
            I_OFF = ioff_real + 1j * ioff_imag
            return A, I_ON, I_OFF

        # 4. Run Optimization
        print("🧠 Running least squares optimization...")
        
        # Initial guess
        A_init = np.array([
            [1,  1,  1,  1],
            [1, 1j, -1,-1j],
            [1, -1,  1, -1],
            [1, -1j, -1, 1j]
        ], dtype=complex) * 0.5
        
        I_ON_init = np.ones(4, dtype=float) * 4.0
        I_OFF_init = np.zeros(4, dtype=complex) + 0.1
        
        x0 = pack(A_init, I_ON_init, I_OFF_init)
        
        res = least_squares(lambda x: compute_residuals(*unpack(x)), x0, method='lm', max_nfev=5000, verbose=1)
        
        A_final, I_ON_final, I_OFF_final = unpack(res.x)
        
        # Normalize
        norm_A = np.linalg.norm(A_final, 2)
        A_final /= norm_A
        I_ON_final *= norm_A
        I_OFF_final *= norm_A
        
        print(f"✅ Optimization complete. Cost: {res.cost:.2e}")
        
        # 5. Plotting
        if plot and plt is not None:
            # Calculate predictions
            Is_ON = I_ON_final[None, :]
            Is_OFF = I_OFF_final[None, :]
            I_base = np.where(active_masks_arr, Is_ON, Is_OFF)
            O_base = (A_final @ I_base.T).T
            k_indices = scanned_indices_arr
            I_k = I_base[np.arange(N_sub), k_indices]
            A_cols = A_final[:, k_indices].T
            O_mod = A_cols * I_k[:, None]
            O_static = O_base - O_mod
            
            pred_dc = np.abs(O_static)**2 + np.abs(O_mod)**2
            pred_fund = O_static * np.conj(O_mod)
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            
            # DC Correlation
            axs[0].scatter(measure_dc_arr.flatten(), pred_dc.flatten(), alpha=0.5, s=10)
            mx_dc = max(measure_dc_arr.max(), pred_dc.max())
            axs[0].plot([0, mx_dc], [0, mx_dc], 'r--')
            axs[0].set_xlabel("Measured DC")
            axs[0].set_ylabel("Predicted DC")
            axs[0].set_title(f"DC Component (RMS Error: {np.sqrt(np.mean((measure_dc_arr-pred_dc)**2)):.2e})")
            
            # Fundamental Correlation (Amplitude)
            axs[1].scatter(np.abs(measure_fund_arr).flatten(), np.abs(pred_fund).flatten(), alpha=0.5, s=10)
            mx_fund = max(np.abs(measure_fund_arr).max(), np.abs(pred_fund).max())
            axs[1].plot([0, mx_fund], [0, mx_fund], 'r--')
            axs[1].set_xlabel("Measured Fund Amp")
            axs[1].set_ylabel("Predicted Fund Amp")
            axs[1].set_title(f"Fundamental (RMS Error: {np.sqrt(np.mean(np.abs(measure_fund_arr-pred_fund)**2)):.2e})")
            
            fig.suptitle(f"Model Fit: {self.name} (Cost: {res.cost:.2e})")
            plt.tight_layout()
            
            if save_as:
                plt.savefig(save_as, dpi=150, bbox_inches='tight')
            plt.show()
            
        return {
            'A': A_final,
            'I_ON': I_ON_final,
            'I_OFF': I_OFF_final,
            'cost': res.cost
        }