# scripts/stage4_supervised/models.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def build_nb_model():
    """
    Naive Bayes for text data (BM25 features – non-negative).
    """
    return MultinomialNB()


def build_logreg_model():
    """
    Logistic Regression classifier for binary classification (UK vs US).
    """
    return LogisticRegression(
        max_iter=1000,
        solver="liblinear",       # מתאים לבעיות sparse ודו-ממדיות
        class_weight="balanced",
        random_state=42
    )


def build_svm_model():
    """
    Linear SVM (LinearSVC) – מאפשר לקבל coef_ למשקלי פיצ'רים.
    """
    return LinearSVC(
        class_weight="balanced",
        random_state=42
    )


def build_rf_model():
    """
    Random Forest classifier – עובד על מטריצה צפופה (לא sparse),
    לכן ב-run_supervised נהפוך את X ל-numpy array לפני השימוש.
    """
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
