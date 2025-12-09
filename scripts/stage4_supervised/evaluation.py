# scripts/stage4_supervised/evaluation.py

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def evaluate_with_cv(model, X, y, n_splits: int = 10):
    """
    מריץ n_splits-fold Stratified Cross Validation על מודל נתון
    ומחזיר ממוצעים של precision, recall, f1, accuracy.

    Parameters
    ----------
    model : sklearn estimator (עם fit/predict)
    X     : features (sparse או dense)
    y     : labels (וקטור 1D)
    n_splits : מספר folds (ברירת מחדל 10)

    Returns
    -------
    dict עם המפתחות:
        "precision", "recall", "f1", "accuracy"
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    precisions, recalls, f1s, accs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        accs.append(accuracy_score(y_test, y_pred))

    return {
        "precision": float(np.mean(precisions)),
        "recall":    float(np.mean(recalls)),
        "f1":        float(np.mean(f1s)),
        "accuracy":  float(np.mean(accs)),
    }
