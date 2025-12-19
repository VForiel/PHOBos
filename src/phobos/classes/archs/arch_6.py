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
    
    def calibrate_obs(
        self,
        dm_object,
        cred3_object,
        crop_centers,
        crop_sizes=10,
        n: int = 1_000,
        n_averages: int = 10,
        plot: bool = False,
        figsize: tuple = (16, 12),
        save_as=None,
    ):
        """
        Optimize calibration via least squares sampling for Architecture 6.
        
        This method systematically scans the 4 phase shifters to optimize the MMI performance.
        With only 4 shifters and 4 outputs, the calibration is simpler than Architecture 15:
        1. Maximize each output independently using one shifter each
        2. Or optimize for specific interference patterns
        
        Parameters
        ----------
        dm_object : DM
            Deformable mirror instance to control input blocking/unblocking.
        cred3_object : Cred3
            Camera instance for flux measurements.
        crop_centers : array-like
            Output spot centers for flux extraction (4 outputs expected).
        crop_sizes : int, optional
            Crop window size. Default is 10.
        n : int, optional
            Number of sampling points for least squares. Default is 1000.
        n_averages : int, optional
            Number of frames to average per phase point. Default is 10.
        plot : bool, optional
            If True, plot the optimization process. Default is False.
        figsize : tuple, optional
            Figure size for plots. Default is (16, 12).
        save_as : str, optional
            Path to save the plot if plot is True. Default is None.
            
        Notes
        -----
        For a 4x4 MMI with 4 shifters, the calibration strategy is simplified:
        - Each shifter is scanned to optimize its corresponding output
        - Different input configurations can be tested
        - The goal is to achieve desired beam combining ratios
        
        Examples
        --------
        >>> from kbench import DM
        >>> from phobos import Arch6
        >>> from phobos.classes.cred3 import Cred3
        >>> 
        >>> arch = Arch6()
        >>> dm = DM()
        >>> cam = Cred3()
        >>> crop_centers = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  # 4 outputs
        >>> 
        >>> arch.calibrate_obs(dm, cam, crop_centers, plot=True)
        """
        
        if len(crop_centers) != 4:
            raise ValueError(f"❌ Architecture 6 expects 4 output spots, got {len(crop_centers)}")
        
        # Wavelength for phase scanning (assume 1550 nm for photonics)
        λ = 1550.0  # nm
        
        if plot and plt is not None:
            _, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
            axs = axs.flatten()
        
        def optimize_output(shifter_idx, output_idx, maximize=True, plt_coords=None):
            """
            Optimize a specific output by scanning a shifter.
            
            Parameters
            ----------
            shifter_idx : int
                Index of shifter to scan (1-4).
            output_idx : int
                Index of output to optimize (0-3).
            maximize : bool
                If True, maximize output; if False, minimize.
            plt_coords : int, optional
                Subplot index for plotting.
            """
            shifter = self.channels[shifter_idx - 1]
            
            x = np.linspace(0, λ, n)
            y = np.empty(n)
            
            for i in range(n):
                shifter.set_phase(2 * np.pi * x[i] / λ)
                
                # Measure target output
                temp_outs = []
                for _ in range(n_averages):
                    outs = cred3_object.get_outputs(crop_centers=crop_centers, crop_sizes=crop_sizes)
                    temp_outs.append(outs[output_idx])
                y[i] = np.mean(temp_outs)
            
            # Normalize
            y_norm = y / np.max(y) if np.max(y) > 0 else y
            
            # Fit sinusoid
            def sin_model(x, x0):
                return (np.sin((x - x0) / λ * 2 * np.pi) + 1) / 2 * (np.max(y_norm) - np.min(y_norm)) + np.min(y_norm)
            
            popt = scipy.optimize.minimize(
                lambda x0: np.sum((y_norm - sin_model(x, x0)) ** 2), 
                x0=[0], 
                method='Nelder-Mead'
            ).x
            
            # Set optimal phase
            if maximize:
                optimal_phase = np.mod(popt[0] + λ / 4, λ)  # π/2 shift for maximum
            else:
                optimal_phase = np.mod(popt[0], λ)  # Minimum
            
            shifter.set_phase(2 * np.pi * optimal_phase / λ)
            
            if plot and plt is not None and plt_coords is not None:
                action = "Maximize" if maximize else "Minimize"
                axs[plt_coords].set_title(f"{action} Output {output_idx + 1} (Shifter {shifter_idx})")
                axs[plt_coords].scatter(x, y_norm, label='Data', color='tab:blue', s=1, alpha=0.5)
                axs[plt_coords].plot(x, sin_model(x, *popt), label='Fit', color='tab:orange', linewidth=2)
                axs[plt_coords].axvline(x=optimal_phase, color='k', linestyle='--', label='Optimal', linewidth=1.5)
                axs[plt_coords].set_xlabel("Phase shift (nm)")
                axs[plt_coords].set_ylabel(f"Output {output_idx + 1} throughput")
                axs[plt_coords].legend()
                axs[plt_coords].grid(True, alpha=0.3)
        
        # ============ Calibration sequence ============
        
        print("🔧 Starting Architecture 6 calibration...")
        
        # Strategy 1: Maximize each output with all inputs active
        # This provides a baseline calibration
        
        dm_object.max()  # All inputs active
        
        print("  [1/4] Optimizing output 1 with shifter 1")
        optimize_output(1, 0, maximize=True, plt_coords=0)
        
        print("  [2/4] Optimizing output 2 with shifter 2")
        optimize_output(2, 1, maximize=True, plt_coords=1)
        
        print("  [3/4] Optimizing output 3 with shifter 3")
        optimize_output(3, 2, maximize=True, plt_coords=2)
        
        print("  [4/4] Optimizing output 4 with shifter 4")
        optimize_output(4, 3, maximize=True, plt_coords=3)
        
        # Alternative strategies could be implemented:
        # - Optimize for specific interference patterns
        # - Maximize contrast between outputs
        # - Minimize specific outputs for nulling
        
        if plot and plt is not None:
            if save_as:
                plt.savefig(save_as, dpi=150, bbox_inches='tight')
            plt.show()
        
        print("✅ Architecture 6 calibration complete!")
    
    def calibrate_nulling(
        self,
        dm_object,
        cred3_object,
        crop_centers,
        bright_output_idx=0,
        null_output_idx=1,
        crop_sizes=10,
        n: int = 500,
        n_averages: int = 10,
        plot: bool = False,
        save_as=None,
    ):
        """
        Alternative calibration strategy for nulling interferometry.
        
        This method optimizes for:
        1. Maximizing one output (bright port)
        2. Minimizing another output (null port)
        
        Parameters
        ----------
        dm_object : DM
            Deformable mirror instance.
        cred3_object : Cred3
            Camera instance.
        crop_centers : array-like
            Output spot centers (4 outputs).
        bright_output_idx : int, optional
            Index of output to maximize. Default is 0.
        null_output_idx : int, optional
            Index of output to minimize. Default is 1.
        crop_sizes : int, optional
            Crop window size. Default is 10.
        n : int, optional
            Sampling points. Default is 500.
        n_averages : int, optional
            Frames to average. Default is 10.
        plot : bool, optional
            If True, plot results. Default is False.
        save_as : str, optional
            Save path for plot.
            
        Examples
        --------
        >>> arch = Arch6()
        >>> arch.calibrate_nulling(dm, cam, crop_centers, 
        ...                        bright_output_idx=0, null_output_idx=1,
        ...                        plot=True)
        """
        
        if len(crop_centers) != 4:
            raise ValueError(f"❌ Architecture 6 expects 4 output spots, got {len(crop_centers)}")
        
        print("🔧 Starting Architecture 6 nulling calibration...")
        print(f"   Target: Maximize output {bright_output_idx + 1}, Minimize output {null_output_idx + 1}")
        
        λ = 1550.0  # nm
        
        # Activate all inputs
        dm_object.max()
        
        # Step 1: Maximize bright output with shifter 1
        print(f"  [1/2] Maximizing output {bright_output_idx + 1}...")
        shifter1 = self.channels[0]
        x = np.linspace(0, λ, n)
        y_bright = np.empty(n)
        
        for i in range(n):
            shifter1.set_phase(2 * np.pi * x[i] / λ)
            temp = []
            for _ in range(n_averages):
                outs = cred3_object.get_outputs(crop_centers=crop_centers, crop_sizes=crop_sizes)
                temp.append(outs[bright_output_idx])
            y_bright[i] = np.mean(temp)
        
        # Fit and set optimal
        y_bright_norm = y_bright / np.max(y_bright)
        def sin_model(x, x0):
            return (np.sin((x - x0) / λ * 2 * np.pi) + 1) / 2 * (np.max(y_bright_norm) - np.min(y_bright_norm)) + np.min(y_bright_norm)
        
        popt_bright = scipy.optimize.minimize(
            lambda x0: np.sum((y_bright_norm - sin_model(x, x0)) ** 2),
            x0=[0],
            method='Nelder-Mead'
        ).x
        
        optimal_bright = np.mod(popt_bright[0] + λ / 4, λ)
        shifter1.set_phase(2 * np.pi * optimal_bright / λ)
        
        # Step 2: Minimize null output with shifter 2
        print(f"  [2/2] Minimizing output {null_output_idx + 1}...")
        shifter2 = self.channels[1]
        y_null = np.empty(n)
        
        for i in range(n):
            shifter2.set_phase(2 * np.pi * x[i] / λ)
            temp = []
            for _ in range(n_averages):
                outs = cred3_object.get_outputs(crop_centers=crop_centers, crop_sizes=crop_sizes)
                temp.append(outs[null_output_idx])
            y_null[i] = np.mean(temp)
        
        # Fit and set optimal (minimum)
        y_null_norm = y_null / np.max(y_null) if np.max(y_null) > 0 else y_null
        popt_null = scipy.optimize.minimize(
            lambda x0: np.sum((y_null_norm - sin_model(x, x0)) ** 2),
            x0=[0],
            method='Nelder-Mead'
        ).x
        
        optimal_null = np.mod(popt_null[0], λ)
        shifter2.set_phase(2 * np.pi * optimal_null / λ)
        
        if plot and plt is not None:
            _, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            
            axs[0].scatter(x, y_bright_norm, label='Data', color='tab:blue', s=1, alpha=0.5)
            axs[0].plot(x, sin_model(x, *popt_bright), label='Fit', color='tab:orange', linewidth=2)
            axs[0].axvline(x=optimal_bright, color='k', linestyle='--', label='Optimal')
            axs[0].set_title(f"Maximize Bright Output {bright_output_idx + 1}")
            axs[0].set_xlabel("Phase shift (nm)")
            axs[0].set_ylabel("Normalized flux")
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            axs[1].scatter(x, y_null_norm, label='Data', color='tab:red', s=1, alpha=0.5)
            axs[1].plot(x, sin_model(x, *popt_null), label='Fit', color='tab:orange', linewidth=2)
            axs[1].axvline(x=optimal_null, color='k', linestyle='--', label='Optimal')
            axs[1].set_title(f"Minimize Null Output {null_output_idx + 1}")
            axs[1].set_xlabel("Phase shift (nm)")
            axs[1].set_ylabel("Normalized flux")
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            if save_as:
                plt.savefig(save_as, dpi=150, bbox_inches='tight')
            plt.show()
        
        print("✅ Nulling calibration complete!")
