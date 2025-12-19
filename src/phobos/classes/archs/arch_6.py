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