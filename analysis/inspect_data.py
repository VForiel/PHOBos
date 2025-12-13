import numpy as np
import os

data_path = r"E:\PHOBos\tests\generated\architecture_characterization\4-Port_MMI_Active\20251208_164250\characterization_data.npz"

if not os.path.exists(data_path):
    print(f"File not found: {data_path}")
    exit(1)

data = np.load(data_path)
print("Keys in npz file:")
for key in data.keys():
    print(f"{key}: {data[key].shape}, dtype={data[key].dtype}")
    # Print a small sample if it's small enough
    if data[key].size < 20:
        print(f"  Values: {data[key]}")
