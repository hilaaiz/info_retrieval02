import os
import shutil
import pandas as pd
import numpy as np
import re
import html

# -------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Original UK/US folders
uk_folder = os.path.join(ROOT_DIR, "UK_british_debates_text_files_normalize")
us_folder = os.path.join(ROOT_DIR, "allData_cleaned")

# Unified folder (output)
output_folder = os.path.join(ROOT_DIR, "allData")
os.makedirs(output_folder, exist_ok=True)


# -------------------------------------------------------------
# CLEANING (you asked not to use it now, but we keep function ready)
# -------------------------------------------------------------
def clean_congressional_text(raw_text):
    """
    CLEANING IS DISABLED FOR NOW.
    We simply return the raw text unchanged.
    """
    return raw_text


# -------------------------------------------------------------
# COPY & RENAME FILES INTO allData/
# -------------------------------------------------------------
def copy_and_rename(src_folder, prefix):
    print(f"Scanning folder: {src_folder}")

    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)

        if not os.path.isfile(src_path):
            continue  # skip subfolders

        # New filename with prefix
        if filename.startswith(prefix + "_"):
            new_name = filename  # do NOT prefix twice
        else:
            new_name = f"{prefix}_{filename}"

        dst_path = os.path.join(output_folder, new_name)

        shutil.copy(src_path, dst_path)
        print("Copied:", new_name)


print("\n=== COPYING FILES TO allData/ ===")
copy_and_rename(uk_folder, "UK")
copy_and_rename(us_folder, "US")
print("✓ Merge complete!\n")


# -------------------------------------------------------------
# CREATE DATASET (metadata + labels)
# -------------------------------------------------------------
print("=== BUILDING DATASET (metadata + labels) ===")

rows = []

for filename in sorted(os.listdir(output_folder)):
    file_path = os.path.join(output_folder, filename)

    if not os.path.isfile(file_path):
        continue

    # Determine label from filename prefix
    if filename.startswith("UK_"):
        country = "UK"
    elif filename.startswith("US_"):
        country = "US"
    else:
        print("Skipping unknown file:", filename)
        continue

    # Load raw text
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    # NO cleaning is applied now
    text = raw_text

    rows.append({
        "text": text,
        "country": country,
        "filename": filename,
    })

# Build DataFrame
df = pd.DataFrame(rows)
df["row_index"] = df.index

print("\nDataset created:")
print(df.head())
print(df.country.value_counts())
print(f"Total documents: {len(df)}")


# -------------------------------------------------------------
# SAVE METADATA + LABEL FILES
# -------------------------------------------------------------
print("\n=== SAVING OUTPUT FILES ===")

metadata_path = os.path.join(ROOT_DIR, "documents_metadata.csv")
df.to_csv(metadata_path, index=False)

label_map = {"UK": 0, "US": 1}
y_num = df["country"].map(label_map).to_numpy()
y_str = df["country"].to_numpy()

np.save(os.path.join(ROOT_DIR, "y_labels_num.npy"), y_num)
np.save(os.path.join(ROOT_DIR, "y_labels_str.npy"), y_str)

print("Saved:")
print(" - documents_metadata.csv")
print(" - y_labels_num.npy")
print(" - y_labels_str.npy")

print("\n✓ STAGE 1 COMPLETE.\n")
