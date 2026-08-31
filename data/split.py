import os
import random
import shutil

random.seed(42)

# INPUT FOLDERS
wind_folder = "combined_latents/wind_complete"
water_folder = "combined_latents/water_complete"

# OUTPUT ROOT
output_root = "ptmodel/dataset"

splits = ["train", "val", "test"]
classes = ["wind", "water"]

# Create folders
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)


def split_files(files):
    random.shuffle(files)

    n = len(files)

    train_end = int(0.80 * n)
    val_end = train_end + int(0.10 * n)

    train = files[:train_end]
    val = files[train_end:val_end]
    test = files[val_end:]

    return train, val, test


# ---------- WIND ----------
wind_files = [f for f in os.listdir(wind_folder) if f.endswith(".pt")]

train, val, test = split_files(wind_files)

for f in train:
    shutil.copy2(
        os.path.join(wind_folder, f),
        os.path.join(output_root, "train", "wind", f)
    )

for f in val:
    shutil.copy2(
        os.path.join(wind_folder, f),
        os.path.join(output_root, "val", "wind", f)
    )

for f in test:
    shutil.copy2(
        os.path.join(wind_folder, f),
        os.path.join(output_root, "test", "wind", f)
    )


# ---------- WATER ----------
water_files = [f for f in os.listdir(water_folder) if f.endswith(".pt")]

train, val, test = split_files(water_files)

for f in train:
    shutil.copy2(
        os.path.join(water_folder, f),
        os.path.join(output_root, "train", "water", f)
    )

for f in val:
    shutil.copy2(
        os.path.join(water_folder, f),
        os.path.join(output_root, "val", "water", f)
    )

for f in test:
    shutil.copy2(
        os.path.join(water_folder, f),
        os.path.join(output_root, "test", "water", f)
    )

print("Dataset split complete.")