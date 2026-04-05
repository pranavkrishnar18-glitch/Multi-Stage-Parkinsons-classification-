import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Path to your sample folder
folder_path = "data/sample"

# Load all DICOM slices
slices = []
for file in os.listdir(folder_path):
    if file.endswith(".dcm"):
        path = os.path.join(folder_path, file)
        ds = pydicom.dcmread(path)
        slices.append(ds.pixel_array)

# Convert to numpy array
volume = np.stack(slices)

print("Original Shape:", volume.shape)

# 🔥 Fix 4D → 3D if needed
if len(volume.shape) == 4:
    volume = volume.squeeze()

print("Fixed Shape:", volume.shape)

# Normalize
volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

# Resize to manageable shape
desired_shape = (64, 128, 128)
factors = [d/s for d, s in zip(desired_shape, volume.shape)]
volume = zoom(volume, factors)

print("Resized Shape:", volume.shape)

# Show middle slice
plt.imshow(volume[volume.shape[0]//2], cmap="gray")
plt.title("Processed Slice")
plt.show()

# Save processed volume
np.save("processed/sample.npy", volume)

print("Saved successfully!")