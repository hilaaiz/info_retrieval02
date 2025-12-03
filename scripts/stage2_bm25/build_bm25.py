"""
Step 1: Build Shared BM25 Matrix for UK + US
============================================

This script:
- Loads all UK and US text files
- Builds ONE shared BM25 matrix (UK + US)
- Creates:
    - X: BM25 matrix (docs x terms)
    - y: labels (UK / US)
    - DataFrame with [text, country, filename, row_index]
    - vocabulary (feature names)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# NLTK stopwords
import nltk
from nltk.corpus import stopwords


# ----------------------------------------------------
# BM25 Transformer (copied from previous exercise)
# ----------------------------------------------------
class BM25Transformer:
    """
    BM25/Okapi Transformer
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit_transform(self, tf_matrix, doc_lengths, avg_doc_length, idf_vector):
        bm25_matrix = tf_matrix.copy()

        for i in range(bm25_matrix.shape[0]):
            doc_len = doc_lengths[i]
            length_norm = 1 - self.b + self.b * (doc_len / avg_doc_length)

            row = bm25_matrix.getrow(i)
            row_data = row.data

            row_data = row_data * (self.k1 + 1) / (row_data + self.k1 * length_norm)

            col_indices = row.indices
            row_data = row_data * idf_vector[col_indices]

            bm25_matrix.data[bm25_matrix.indptr[i]:bm25_matrix.indptr[i+1]] = row_data

        return bm25_matrix


# ----------------------------------------------------
# NLTK stopwords helpers
# ----------------------------------------------------
def download_nltk_data():
    print("\n📥 Checking NLTK stopwords...")
    try:
        _ = stopwords.words("english")
        print("✅ NLTK stopwords already available")
    except LookupError:
        print("📥 Downloading NLTK stopwords...")
        nltk.download("stopwords", quiet=True)
        print("✅ Download completed!")


def get_nltk_stopwords():
    print("\n🛑 Loading NLTK stopwords...")
    download_nltk_data()
    sw = set(stopwords.words("english"))
    print(f"   • Loaded {len(sw)} stopwords (pure NLTK)")
    return sw


# ----------------------------------------------------
# Load UK / US documents
# ----------------------------------------------------
def load_country_documents(uk_folder, us_folder):
    """
    Reads all .txt files from UK and US folders.
    Returns a DataFrame with: text, country, filename
    """

    def load_from_folder(folder_path, country_label):
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        print(f"\n📂 Loading {country_label} documents from: {folder}")
        txt_files = sorted(list(folder.glob("*.txt")))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {folder}")

        rows = []
        for txt_file in tqdm(txt_files, desc=f"Loading {country_label} files"):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"⚠️ Error reading {txt_file.name}: {e}")
                text = ""

            if text.strip():
                rows.append({
                    "text": text,
                    "country": country_label,
                    "filename": txt_file.name  # or txt_file.stem if you prefer
                })

        return rows

    uk_rows = load_from_folder(uk_folder, "UK")
    us_rows = load_from_folder(us_folder, "US")

    df = pd.DataFrame(uk_rows + us_rows)
    print(f"\n✅ Total documents loaded: {len(df)}")
    print(df["country"].value_counts())
    return df


# ----------------------------------------------------
# Build TF-IDF + BM25 on ALL documents together
# ----------------------------------------------------
def build_bm25_matrix(documents, stopwords_set,
                      min_df=5, max_df=0.95, max_features=20000,
                      matrix_name="BM25-UK-US"):
    """
    One shared vectorizer for UK+US.
    """

    print(f"\n{'='*70}")
    print(f"🔨 Building {matrix_name}")
    print(f"{'='*70}")

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words=list(stopwords_set),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 1),
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )

    print("\n🔄 Fitting TF-IDF vectorizer on ALL documents (UK+US)...")
    tfidf_matrix = vectorizer.fit_transform(tqdm(documents, desc="Vectorizing"))
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n✅ TF-IDF created: shape={tfidf_matrix.shape}")

    # BM25
    print("\n🔄 Applying BM25 transformation...")
    doc_lengths = np.array(tfidf_matrix.sum(axis=1)).flatten()
    avg_doc_length = doc_lengths.mean()
    idf_vector = vectorizer.idf_

    bm25_matrix = BM25Transformer().fit_transform(
        tfidf_matrix, doc_lengths, avg_doc_length, idf_vector
    )

    stats = {
        "matrix_name": matrix_name,
        "num_documents": bm25_matrix.shape[0],
        "num_features": bm25_matrix.shape[1],
        "sparsity": (1 - bm25_matrix.nnz / (bm25_matrix.shape[0] * bm25_matrix.shape[1])) * 100,
        "non_zero_elements": bm25_matrix.nnz,
    }

    print("✅ BM25 matrix ready")
    print(f"   • Documents: {stats['num_documents']}")
    print(f"   • Features: {stats['num_features']}")
    print(f"   • Sparsity: {stats['sparsity']:.2f}%")

    return bm25_matrix, feature_names, vectorizer, stats


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Step 1: Shared BM25 Matrix for UK + US                     ║
║   (One vocabulary, labels = country, full mapping)           ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # === 1. Default Paths ===
    DEFAULT_UK = "UK_british_debates_text_files_normalize"
    DEFAULT_US = "US_congressional_speeches_Text_Files"
    DEFAULT_OUTPUT = "uk_us_outputs"

    UK_FOLDER = input(f"Enter path to UK folder [{DEFAULT_UK}]: ").strip()
    US_FOLDER = input(f"Enter path to US folder [{DEFAULT_US}]: ").strip()
    OUTPUT_FOLDER = input(f"Enter path for output folder [{DEFAULT_OUTPUT}]: ").strip()

    # If user just pressed Enter → use defaults
    UK_FOLDER = UK_FOLDER if UK_FOLDER else DEFAULT_UK
    US_FOLDER = US_FOLDER if US_FOLDER else DEFAULT_US
    OUTPUT_FOLDER = OUTPUT_FOLDER if OUTPUT_FOLDER else DEFAULT_OUTPUT


   

    # === 2. Stopwords ===
    nltk_stopwords = get_nltk_stopwords()

    # === 3. Load UK + US documents into ONE DataFrame ===
    df = load_country_documents(UK_FOLDER, US_FOLDER)
    df = df.reset_index(drop=True)
    df["row_index"] = df.index  # mapping row -> doc

    # === 4. Build BM25 matrix on ALL documents together ===
    documents = df["text"].tolist()

    BM25_MIN_DF = 5
    BM25_MAX_DF = 0.95
    BM25_MAX_FEATURES = 20000

    X_bm25, feature_names, vectorizer, stats = build_bm25_matrix(
        documents=documents,
        stopwords_set=nltk_stopwords,
        min_df=BM25_MIN_DF,
        max_df=BM25_MAX_DF,
        max_features=BM25_MAX_FEATURES,
        matrix_name="BM25-UK-US"
    )

    # === 5. Create labels vector y ===
    # Option 1: keep as strings "UK"/"US"
    y_str = df["country"].values

    # Option 2: numeric labels 0=UK, 1=US (useful for sklearn)
    label_map = {"UK": 0, "US": 1}
    y_num = df["country"].map(label_map).values

    # === 6. Save everything ===
    print("\n💾 Saving outputs...")

    # BM25 matrix
    save_npz(OUTPUT_FOLDER / "X_bm25_uk_us.npz", X_bm25)

    # labels
    np.save(OUTPUT_FOLDER / "y_labels_str.npy", y_str)
    np.save(OUTPUT_FOLDER / "y_labels_num.npy", y_num)

    # feature names (vocabulary)
    with open(OUTPUT_FOLDER / "bm25_feature_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))

    # DataFrame mapping (text + metadata)
    df.to_csv(OUTPUT_FOLDER / "documents_metadata.csv", index=False)

    # Stats
    pd.DataFrame([stats]).to_csv(OUTPUT_FOLDER / "bm25_stats.csv", index=False)

    print("\n🎉 Done!")
    print(f"   • X matrix: {OUTPUT_FOLDER / 'X_bm25_uk_us.npz'}")
    print(f"   • y (str):  {OUTPUT_FOLDER / 'y_labels_str.npy'}")
    print(f"   • y (num):  {OUTPUT_FOLDER / 'y_labels_num.npy'}")
    print(f"   • metadata: {OUTPUT_FOLDER / 'documents_metadata.csv'}")
    print(f"   • vocab:    {OUTPUT_FOLDER / 'bm25_feature_names.txt'}")


if __name__ == "__main__":
    main()
