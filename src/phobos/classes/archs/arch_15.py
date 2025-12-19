from ..photonic_chip import Arch
import numpy as np
import scipy.optimize

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class Arch15(Arch):
    """
    Architecture 15: Mega Kernel Nuller Reconfig (14 shifters, 4 inputs, 7 outputs).
    
    This is the full kernel-nulling architecture with:
    - 4 inputs (controlled via DM)
    - 7 outputs: 1 Bright + 6 Darks + (3 Kernels computed from Darks)
    - 14 phase shifters (TOPAs)
    """
    
    def __init__(self):
        super().__init__(
            name="Mega Kernel Nuller Reconfig",
            id="N2x2-D4",
            n_inputs=4,
            n_outputs=7,
            topas=(6,7,33,34,35,36,37,38,28,27,26,25,39,40),
            number=15
        )
    
    def calibrate_obs(
        self,
        dm_object,
        cred3_object,
        crop_centers,
        crop_sizes=10,
        n: int = 1_000,
        n_averages: int = 10,
        plot: bool = False,
        figsize: tuple = (30, 20),
        save_as=None,
    ):
        """
        Optimize calibration via least squares sampling for Architecture 15.
        
        This method systematically scans phase shifters to optimize the kernel nuller performance:
        1. Maximize bright output (3 configurations)
        2. Maximize dark pairs (1 configuration)
        3. Minimize kernel outputs (3 configurations)
        
        Parameters
        ----------
        dm_object : DM
            Deformable mirror instance to control input blocking/unblocking.
        cred3_object : Cred3
            Camera instance for flux measurements.
        crop_centers : array-like
            Output spot centers for flux extraction (7 outputs expected).
        crop_sizes : int, optional
            Crop window size. Default is 10.
        n : int, optional
            Number of sampling points for least squares. Default is 1000.
        n_averages : int, optional
            Number of frames to average per phase point. Default is 10.
        plot : bool, optional
            If True, plot the optimization process. Default is False.
        figsize : tuple, optional
            Figure size for plots. Default is (30, 20).
        save_as : str, optional
            Path to save the plot if plot is True. Default is None.
            
        Notes
        -----
        The calibration follows the strategy from the PHISE simulation:
        - Shifters 2, 4, 7 control bright maximization with different input pairs
        - Shifter 8 controls dark pair symmetry
        - Shifters 11, 13, 14 minimize kernel outputs
        - If polychromatic: shifters [1,2], [3,4], [5,7] for achromatic bright control
        
        The method scans each shifter from 0 to 2π (or λ in length units), fits a sinusoid,
        and sets the optimal phase for maximum/minimum transmission.
        
        Examples
        --------
        >>> from kbench import DM
        >>> from phobos import Arch15
        >>> from phobos.classes.cred3 import Cred3
        >>> 
        >>> arch = Arch15()
        >>> dm = DM()
        >>> cam = Cred3()
        >>> crop_centers = [[x1, y1], [x2, y2], ..., [x7, y7]]  # 7 outputs
        >>> 
        >>> arch.calibrate_obs(dm, cam, crop_centers, plot=True)
        """
        
        if len(crop_centers) != 7:
            raise ValueError(f"❌ Architecture 15 expects 7 output spots, got {len(crop_centers)}")
        
        # Wavelength for phase scanning (assume 1550 nm for photonics)
        λ = 1550.0  # nm
        
        if plot and plt is not None:
            _, axs = plt.subplots(6, 3, figsize=figsize, constrained_layout=True)
            for i in range(7):
                axs.flatten()[i].set_xlabel("Phase shift (nm)")
                axs.flatten()[i].set_ylabel("Throughput (normalized)")
        
        def maximize_bright(shifter_indices, plt_coords=None):
            """Maximize bright output (output 0) by scanning shifter(s)."""
            if not isinstance(shifter_indices, list):
                shifter_indices = [shifter_indices]
            
            # Get the PhaseShifter objects
            shifters = [self.channels[idx - 1] for idx in shifter_indices]
            
            # If multiple shifters, maintain relative phase
            if len(shifters) > 1:
                initial_phases = [ch.get_phase() for ch in shifters]
                Δφ = initial_phases[1] - initial_phases[0] if len(initial_phases) > 1 else 0
            
            x = np.linspace(0, λ, n)
            y = np.empty(n)
            
            for i in range(n):
                # Set phase for primary shifter
                shifters[0].set_phase(2 * np.pi * x[i] / λ)
                
                # If multiple shifters, maintain phase difference
                if len(shifters) > 1:
                    shifters[1].set_phase(2 * np.pi * x[i] / λ + Δφ)
                
                # Measure bright output (index 0)
                temp_outs = []
                for _ in range(n_averages):
                    outs = cred3_object.get_outputs(crop_centers=crop_centers, crop_sizes=crop_sizes)
                    temp_outs.append(outs[0])  # Bright output
                y[i] = np.mean(temp_outs)
            
            # Normalize
            y = y / np.max(y) if np.max(y) > 0 else y
            
            # Fit sinusoid
            def sin_model(x, x0):
                return (np.sin((x - x0) / λ * 2 * np.pi) + 1) / 2 * (np.max(y) - np.min(y)) + np.min(y)
            
            popt = scipy.optimize.minimize(lambda x0: np.sum((y - sin_model(x, x0)) ** 2), x0=[0], method='Nelder-Mead').x
            
            # Set optimal phase (π/2 shift for maximum)
            optimal_phase = np.mod(popt[0] + λ / 4, λ)
            shifters[0].set_phase(2 * np.pi * optimal_phase / λ)
            
            if len(shifters) > 1:
                shifters[1].set_phase(2 * np.pi * optimal_phase / λ + Δφ)
            
            if plot and plt is not None and plt_coords is not None:
                axs[plt_coords].set_title(f"Bright (shifter {'s' if len(shifter_indices) > 1 else ''} {shifter_indices})")
                axs[plt_coords].scatter(x, y, label='Data', color='tab:blue', s=1)
                axs[plt_coords].plot(x, sin_model(x, *popt), label='Fit', color='tab:orange')
                axs[plt_coords].axvline(x=optimal_phase, color='k', linestyle='--', label='Optimal')
                axs[plt_coords].set_xlabel("Phase shift (nm)")
                axs[plt_coords].set_ylabel("Bright throughput")
                axs[plt_coords].legend()
        
        def minimize_kernel(shifter_idx, kernel_idx, plt_coords=None):
            """Minimize kernel output by scanning a shifter."""
            shifter = self.channels[shifter_idx - 1]
            
            x = np.linspace(0, λ, n)
            y = np.empty(n)
            
            for i in range(n):
                shifter.set_phase(2 * np.pi * x[i] / λ)
                
                # Measure outputs and compute kernel
                temp_outs = []
                for _ in range(n_averages):
                    outs = cred3_object.get_outputs(crop_centers=crop_centers, crop_sizes=crop_sizes)
                    # Kernel = |D_{2k-1}|² - |D_{2k}|²
                    # Outputs: [Bright, D1, D2, D3, D4, D5, D6]
                    # Kernels: K1 = D1 - D2, K2 = D3 - D4, K3 = D5 - D6
                    if kernel_idx == 1:
                        kernel = outs[1] - outs[2]
                    elif kernel_idx == 2:
                        kernel = outs[3] - outs[4]
                    else:  # kernel_idx == 3
                        kernel = outs[5] - outs[6]
                    temp_outs.append(np.abs(kernel))
                y[i] = np.mean(temp_outs)
            
            # Normalize
            y = y / np.max(y) if np.max(y) > 0 else y
            
            # Fit sinusoid
            def sin_model(x, x0):
                return (np.sin((x - x0) / λ * 2 * np.pi) + 1) / 2 * (np.max(y) - np.min(y)) + np.min(y)
            
            popt = scipy.optimize.minimize(lambda x0: np.sum((y - sin_model(x, x0)) ** 2), x0=[0], method='Nelder-Mead').x
            
            # Set optimal phase (minimum)
            optimal_phase = np.mod(popt[0], λ)
            shifter.set_phase(2 * np.pi * optimal_phase / λ)
            
            if plot and plt is not None and plt_coords is not None:
                axs[plt_coords].set_title(f"Kernel {kernel_idx} (shifter {shifter_idx})")
                axs[plt_coords].scatter(x, y, label='Data', color='tab:blue', s=1)
                axs[plt_coords].plot(x, sin_model(x, *popt), label='Fit', color='tab:orange')
                axs[plt_coords].axvline(x=optimal_phase, color='k', linestyle='--', label='Optimal')
                axs[plt_coords].set_xlabel("Phase shift (nm)")
                axs[plt_coords].set_ylabel(f"K{kernel_idx} throughput")
                axs[plt_coords].legend()
        
        def maximize_darks(shifter_idx, dark_indices, plt_coords=None):
            """Maximize sum of dark pair outputs."""
            shifter = self.channels[shifter_idx - 1]
            
            x = np.linspace(0, λ, n)
            y = np.empty(n)
            
            for i in range(n):
                shifter.set_phase(2 * np.pi * x[i] / λ)
                
                # Measure dark pair sum
                temp_outs = []
                for _ in range(n_averages):
                    outs = cred3_object.get_outputs(crop_centers=crop_centers, crop_sizes=crop_sizes)
                    # Sum specified dark outputs (1-indexed to 0-indexed)
                    dark_sum = np.sum([outs[d] for d in dark_indices])
                    temp_outs.append(dark_sum)
                y[i] = np.mean(temp_outs)
            
            # Normalize
            y = y / np.max(y) if np.max(y) > 0 else y
            
            # Fit sinusoid
            def sin_model(x, x0):
                return (np.sin((x - x0) / λ * 2 * np.pi) + 1) / 2 * (np.max(y) - np.min(y)) + np.min(y)
            
            popt = scipy.optimize.minimize(lambda x0: np.sum((y - sin_model(x, x0)) ** 2), x0=[0], method='Nelder-Mead').x
            
            # Set optimal phase (π/2 shift for maximum)
            optimal_phase = np.mod(popt[0] + λ / 4, λ)
            shifter.set_phase(2 * np.pi * optimal_phase / λ)
            
            if plot and plt is not None and plt_coords is not None:
                axs[plt_coords].set_title(f"Darks {dark_indices} (shifter {shifter_idx})")
                axs[plt_coords].scatter(x, y, label='Data', color='tab:blue', s=1)
                axs[plt_coords].plot(x, sin_model(x, *popt), label='Fit', color='tab:orange')
                axs[plt_coords].axvline(x=optimal_phase, color='k', linestyle='--', label='Optimal')
                axs[plt_coords].set_xlabel("Phase shift (nm)")
                axs[plt_coords].set_ylabel(f"Dark pair throughput")
                axs[plt_coords].legend()
        
        # ============ Calibration sequence ============
        
        print("🔧 Starting Architecture 15 calibration...")
        
        # Bright maximization (single shifters with different input pairs)
        print("  [1/7] Maximizing bright with inputs 1,2 → shifter 2")
        dm_object.off()
        dm_object.max([1, 2])
        maximize_bright(2, plt_coords=(0, 0))
        
        print("  [2/7] Maximizing bright with inputs 3,4 → shifter 4")
        dm_object.off()
        dm_object.max([3, 4])
        maximize_bright(4, plt_coords=(0, 1))
        
        print("  [3/7] Maximizing bright with inputs 1,3 → shifter 7")
        dm_object.off()
        dm_object.max([1, 3])
        maximize_bright(7, plt_coords=(0, 2))
        
        # Darks maximization
        print("  [4/7] Maximizing dark pair [1,2] with inputs 1,4 (inverted) → shifter 8")
        dm_object.off()
        dm_object.max([1, 4])  # Note: In simulation, input 4 is inverted (attenuation=-1)
        maximize_darks(8, [1, 2], plt_coords=(1, 0))
        
        # Kernel minimization (requires all inputs active)
        print("  [5/7] Minimizing kernel 1 with input 1 only → shifter 11")
        dm_object.off()
        dm_object.max([1])
        minimize_kernel(11, 1, plt_coords=(2, 0))
        
        print("  [6/7] Minimizing kernel 2 with input 1 only → shifter 13")
        minimize_kernel(13, 2, plt_coords=(2, 1))
        
        print("  [7/7] Minimizing kernel 3 with input 1 only → shifter 14")
        minimize_kernel(14, 3, plt_coords=(2, 2))
        
        # Polychromatic correction (shifter pairs for achromatic control)
        # This would require additional calibration data
        # Skipped for now as it requires knowing if setup is monochromatic or not
        
        # Restore all inputs
        dm_object.max()
        
        if plot and plt is not None:
            axs[1, 1].axis('off')
            axs[1, 2].axis('off')
            
            if save_as:
                plt.savefig(save_as, dpi=150, bbox_inches='tight')
            plt.show()
        
        print("✅ Architecture 15 calibration complete!")
