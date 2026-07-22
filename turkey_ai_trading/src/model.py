import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# output/ папка рядом с main.py (на уровень выше src/)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

NON_FEATURE_COLS = ["Target", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

def train(df: pd.DataFrame):
    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["Target"]

    # Временной split — без перемешивания!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples ...")
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    # Сохраняем модель
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUTPUT_DIR / "gbrt_model.pkl")
    print(f"Model saved to {OUTPUT_DIR / 'gbrt_model.pkl'}")

    # Оцениваем результаты
    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    return clf, X_test, y_test, y_pred