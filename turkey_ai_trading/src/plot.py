import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def plot_equity(df: pd.DataFrame, ticker: str = "THYAO.IS"):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df["Cum_Strategy"], label="AI Strategy", color="#2ecc71", linewidth=2)
    ax.plot(df["Cum_BuyHold"],  label="Buy & Hold",  color="#e74c3c", linewidth=2, linestyle="--")
    ax.set_title(f"Cumulative Returns — {ticker}  (AI vs. Buy & Hold)", fontsize=15)
    ax.set_xlabel("Trading Days (test set)")
    ax.set_ylabel("Equity (x initial capital)")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "cumulative_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_feature_importance(model_path: Path):
    clf = joblib.load(model_path)
    importances = clf.feature_importances_
    features    = clf.feature_names_in_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features[order], importances[order], color="#3498db")
    ax.set_title("Feature Importance (Gradient Boosting)", fontsize=14)
    ax.set_xlabel("Relative Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")