import numpy as np
import os
import sys

# Try loading the inputs
try:
    data = np.load('processed_harmonics.npy', allow_pickle=True)
    print("Successfully loaded processed_harmonics.npy")
    print(f"Shape/Len: {len(data)}")
except Exception as e:
    print(f"Failed to load: {e}")
