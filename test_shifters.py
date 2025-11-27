"""Test phase shifter characterization."""

import sys
sys.path.insert(0, "D:\\Kbench\\src")

from kbench.scripts.N4x4_T8.characterize import run_full_characterization

# Test with phase shifters
run_full_characterization(
    dm=None,
    chip=None,
    use_shifters=True,
    wavelength=1550.0,
    n_steps=30,
    crosstalk=0.0,
    output_dir="generated/N4x4_T8_characterization/",
    verbose=False
)
