# scripts/stage4_supervised/run_supervised.py

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import matplotlib.pyplot as plt

from models import (
    build_nb_model,
    build_logreg_model,
    build_svm_model,
    build_rf_model,
)
from evaluation import evaluate_with_cv


# קבועים – נניח שהמיפוי כמו בשלב BM25: UK -> 0, US -> 1
CLASS_NAMES = {0: "UK", 1: "US"}


def load_data():
    """
    טוען את מטריצת ה-BM25, הלייבלים ושמות הפיצ'רים.
    מניח שהקובץ מופיע בתיקייה ../uk_us_outputs יחסית לסקריפט.
    """
    base = Path(__file__).resolve().parents[2] / "uk_us_outputs"

    X = load_npz(base / "X_bm25_uk_us.npz")
    y = np.load(base / "y_labels_num.npy")

    with open(base / "bm25_feature_names.txt", encoding="utf-8") as f:
        feature_names = np.array([line.strip() for line in f if line.strip()])

    return X, y, feature_names, base


# ==========
# TOP FEATURES HELPERS
# ==========

def extract_top_features_nb(model, feature_names, top_k=20):
    """
    מקבל מודל Naive Bayes מאומן ומחזיר רשימת רשומות
    עם 20 המאפיינים החשובים לכל קלאס לפי feature_log_prob_.
    """
    rows = []
    log_probs = model.feature_log_prob_   # shape: (n_classes, n_features)
    n_classes = log_probs.shape[0]

    for c in range(n_classes):
        class_name = CLASS_NAMES.get(c, f"class_{c}")
        scores = log_probs[c]
        top_idx = np.argsort(scores)[-top_k:][::-1]

        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "model": "NaiveBayes",
                "class": class_name,
                "rank": rank,
                "feature": feature_names[idx],
                "score": float(scores[idx]),
            })
    return rows


def extract_top_features_linear(model, feature_names, model_name, top_k=20):
    """
    מוציא 20 מאפיינים חשובים לכל קלאס ממודל לינארי עם coef_ (LogReg / LinearSVC).

    קלאס 1 – המאפיינים עם המשקל הכי חיובי.
    קלאס 0 – המאפיינים עם המשקל הכי שלילי.
    """
    rows = []
    weights = model.coef_[0]   # shape: (n_features,)

    # class 1 (נניח US) – המשקולות החיוביות הגבוהות ביותר
    top_pos_idx = np.argsort(weights)[-top_k:][::-1]
    # class 0 (נניח UK) – המשקולות השליליות "הכי נמוכות"
    top_neg_idx = np.argsort(weights)[:top_k]

    # קלאס US / 1
    class_name_pos = CLASS_NAMES.get(1, "class_1")
    for rank, idx in enumerate(top_pos_idx, start=1):
        rows.append({
            "model": model_name,
            "class": class_name_pos,
            "rank": rank,
            "feature": feature_names[idx],
            "score": float(weights[idx]),
        })

    # קלאס UK / 0
    class_name_neg = CLASS_NAMES.get(0, "class_0")
    for rank, idx in enumerate(top_neg_idx, start=1):
        rows.append({
            "model": model_name,
            "class": class_name_neg,
            "rank": rank,
            "feature": feature_names[idx],
            "score": float(weights[idx]),
        })

    return rows


def extract_top_features_rf(model, feature_names, top_k=20):
    """
    Random Forest – feature_importances_ גלובלי, לא לפי קלאס.
    נציין class="ALL (global)" ונבחר את 20 המאפיינים החשובים ביותר.
    """
    rows = []
    importances = model.feature_importances_   # shape: (n_features,)

    top_idx = np.argsort(importances)[-top_k:][::-1]

    for rank, idx in enumerate(top_idx, start=1):
        rows.append({
            "model": "RandomForest",
            "class": "ALL (global)",
            "rank": rank,
            "feature": feature_names[idx],
            "score": float(importances[idx]),
        })

    return rows


# ==========
# PLOTTING
# ==========

def plot_scores(metrics_df, output_path):
    """
    מצייר גרף עמודות שמשווה Accuracy ו-F1 בין המודלים
    ושומר אותו כ־PNG.
    """
    models = metrics_df["model"].tolist()
    accuracies = metrics_df["accuracy"].values
    f1_scores = metrics_df["f1"].values

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, accuracies, width, label="Accuracy")
    plt.bar(x + width/2, f1_scores, width, label="F1 Score")

    plt.xticks(x, models, rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Supervised Models – Accuracy & F1 (10-fold CV)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# ==========
# MAIN PIPELINE
# ==========

def main():
    X, y, feature_names, base = load_data()

    print("Loaded BM25 matrix:")
    print(f"  X shape: {X.shape}")
    print(f"  y length: {len(y)}")
    print(f"  num features: {len(feature_names)}\n")

    metrics_rows = []
    feature_rows = []

    # ------------------------
    # 1. Naive Bayes
    # ------------------------
    print("=== Naive Bayes (MultinomialNB) ===")
    nb_model_for_cv = build_nb_model()
    nb_metrics = evaluate_with_cv(nb_model_for_cv, X, y, n_splits=10)
    print("CV metrics:", nb_metrics, "\n")

    metrics_rows.append({
        "model": "NaiveBayes", **nb_metrics
    })

    # מאמנים על כל הדאטה בשביל פיצ'רים חשובים
    nb_model_full = build_nb_model()
    nb_model_full.fit(X, y)
    feature_rows.extend(extract_top_features_nb(nb_model_full, feature_names))

    # ------------------------
    # 2. Logistic Regression
    # ------------------------
    print("=== Logistic Regression ===")
    lr_model_for_cv = build_logreg_model()
    lr_metrics = evaluate_with_cv(lr_model_for_cv, X, y, n_splits=10)
    print("CV metrics:", lr_metrics, "\n")

    metrics_rows.append({
        "model": "LogisticRegression", **lr_metrics
    })

    lr_model_full = build_logreg_model()
    lr_model_full.fit(X, y)
    feature_rows.extend(
        extract_top_features_linear(lr_model_full, feature_names, "LogisticRegression")
    )

    # ------------------------
    # 3. Linear SVM
    # ------------------------
    print("=== Linear SVM (LinearSVC) ===")
    svm_model_for_cv = build_svm_model()
    svm_metrics = evaluate_with_cv(svm_model_for_cv, X, y, n_splits=10)
    print("CV metrics:", svm_metrics, "\n")

    metrics_rows.append({
        "model": "LinearSVM", **svm_metrics
    })

    svm_model_full = build_svm_model()
    svm_model_full.fit(X, y)
    feature_rows.extend(
        extract_top_features_linear(svm_model_full, feature_names, "LinearSVM")
    )

    # ------------------------
    # 4. Random Forest
    # ------------------------
    print("=== Random Forest ===")
    # RandomForest לא תומך ב-sparse → הופכים ל-dense
    X_dense = X.toarray()

    rf_model_for_cv = build_rf_model()
    rf_metrics = evaluate_with_cv(rf_model_for_cv, X_dense, y, n_splits=10)
    print("CV metrics:", rf_metrics, "\n")

    metrics_rows.append({
        "model": "RandomForest", **rf_metrics
    })

    rf_model_full = build_rf_model()
    rf_model_full.fit(X_dense, y)
    feature_rows.extend(extract_top_features_rf(rf_model_full, feature_names))

    # ------------------------
    # SAVE RESULTS
    # ------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    features_df = pd.DataFrame(feature_rows)

    metrics_path = base / "supervised_metrics.csv"
    features_path = base / "supervised_top_features.xlsx"
    plot_path = base / "supervised_scores.png"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    features_df.to_excel(features_path, index=False)

    # גרף השוואת Accuracy & F1
    plot_scores(metrics_df, plot_path)

    print("=== DONE ===")
    print("\nMetrics table:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics to:        {metrics_path}")
    print(f"Saved top features to:   {features_path}")
    print(f"Saved scores plot to:    {plot_path}")


if __name__ == "__main__":
    main()
