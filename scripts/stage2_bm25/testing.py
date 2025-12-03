import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from pathlib import Path

# ------------------------------------------
# Change this to your BM25 output folder:
# ------------------------------------------
BM25_FOLDER = Path("uk_us_outputs")  # update if needed

MATRIX_PATH = BM25_FOLDER / "X_bm25_uk_us.npz"
VOCAB_PATH = BM25_FOLDER / "bm25_feature_names.txt"
META_PATH = BM25_FOLDER / "documents_metadata.csv"
YNUM_PATH = BM25_FOLDER / "y_labels_num.npy"


print("\n===== TESTING STAGE 2 (BM25) =====\n")

# ------------------------------------------
# 1. Check files exist
# ------------------------------------------
required_files = {
    "BM25 Matrix": MATRIX_PATH,
    "Vocabulary": VOCAB_PATH,
    "Metadata": META_PATH,
    "Labels": YNUM_PATH,
}

for name, path in required_files.items():
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {name} at {path}")
    else:
        print(f"✔ Found: {name}")


# ------------------------------------------
# 2. Load metadata
# ------------------------------------------
df = pd.read_csv(META_PATH)
num_docs = len(df)
print(f"\n✔ Metadata loaded: {num_docs} documents")


# ------------------------------------------
# 3. Load BM25 matrix
# ------------------------------------------
X = load_npz(MATRIX_PATH)
print(f"✔ BM25 matrix shape: {X.shape}")

if X.shape[0] != num_docs:
    raise ValueError("❌ BM25 rows do NOT match number of documents!")
else:
    print("✔ BM25 matrix row count matches documents")


# ------------------------------------------
# 4. Load vocabulary
# ------------------------------------------
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = f.read().splitlines()

num_features = len(vocab)
print(f"✔ Vocabulary size: {num_features}")

if X.shape[1] != num_features:
    raise ValueError("❌ BM25 columns do NOT match vocabulary size!")
else:
    print("✔ Matrix feature count matches vocabulary")


# ------------------------------------------
# 5. Check labels
# ------------------------------------------
y = np.load(YNUM_PATH)

print(f"\n✔ Loaded y labels: {len(y)} labels")

if len(y) != num_docs:
    raise ValueError("❌ Label count does NOT match document count!")
else:
    print("✔ Label count matches document count")


# ------------------------------------------
# 6. Check sparsity (expected: > 95%)
# ------------------------------------------
sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))
print(f"\n✔ BM25 sparsity: {sparsity*100:.2f}% (expected: VERY sparse)")


print("\n===== STAGE 2 PASSED SUCCESSFULLY (if no errors above) =====")
