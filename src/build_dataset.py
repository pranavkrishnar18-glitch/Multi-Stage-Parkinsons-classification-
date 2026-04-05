import os
import numpy as np
import pydicom
from scipy.ndimage import zoom

# Load one patient folder
def load_scan(folder_path):
    slices = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                path = os.path.join(root, file)
                ds = pydicom.dcmread(path)
                slices.append(ds.pixel_array)

    if len(slices) == 0:
        return None

    volume = np.stack(slices)

    # Fix shape issues
    if volume.shape[0] == 1:
        volume = np.squeeze(volume, axis=0)

    # Resize to (64,128,128)
    factors = (
        64 / volume.shape[0],
        128 / volume.shape[1],
        128 / volume.shape[2]
    )
    volume = zoom(volume, factors)

    # 🔥 Robust normalization
    volume = np.clip(volume, np.percentile(volume, 1), np.percentile(volume, 99))
    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)

    return volume


def build_dataset(base_path):
    X = []
    y = []

    for label, category in enumerate(["healthy", "parkinson"]):
        category_path = os.path.join(base_path, category)

        for patient in os.listdir(category_path):
            patient_path = os.path.join(category_path, patient)

            print(f"Processing {category.upper()}: {patient_path}")

            volume = load_scan(patient_path)

            if volume is not None:
                X.append(volume)
                y.append(label)
                print("Loaded ✅")
            else:
                print("Skipped ❌")

    return np.array(X), np.array(y)


# 🔥 BUILD TRAIN
print("\n--- BUILDING TRAIN DATA ---")
X_train, y_train = build_dataset("data/train")

# 🔥 BUILD VAL
print("\n--- BUILDING VAL DATA ---")
X_val, y_val = build_dataset("data/val")

# Save
os.makedirs("processed", exist_ok=True)

np.save("processed/X_train.npy", X_train)
np.save("processed/y_train.npy", y_train)

np.save("processed/X_val.npy", X_val)
np.save("processed/y_val.npy", y_val)

print("\nDataset saved!")
print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
