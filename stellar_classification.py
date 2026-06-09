"""
Stellar Classification using Logistic Regression and Decision Tree
Dataset: Stellar Classification Dataset - SDSS17
"""

# ─────────────────────────────────────────────
# 1. Load the appropriate software packages
# ─────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset generation
# Mirrors the Stellar Classification Dataset - SDSS17
# (100 000 observations: GALAXY ≈59%, STAR ≈21%, QSO ≈20%)
# ─────────────────────────────────────────────
def load_or_generate_dataset(csv_path: str) -> pd.DataFrame:
    """Load CSV when present; otherwise synthesise a dataset with realistic SDSS-17 statistics."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # The SDSS17 CSV sometimes ships with an empty header for the alpha (RA) column
        if "Unnamed: 1" in df.columns:
            df = df.rename(columns={"Unnamed: 1": "alpha"})
        return df

    print(f"[INFO] '{csv_path}' not found – generating synthetic SDSS-17 dataset …")
    rng = np.random.default_rng(seed=42)
    n = 10_000
    # Class proportions
    classes = rng.choice(["GALAXY", "STAR", "QSO"], size=n, p=[0.594, 0.211, 0.195])

    rows = []
    for cls in classes:
        if cls == "GALAXY":
            u, g = rng.normal(19.5, 1.5), rng.normal(18.5, 1.4)
            r, i, z = rng.normal(17.8, 1.3), rng.normal(17.4, 1.3), rng.normal(17.1, 1.3)
            redshift = rng.exponential(0.35)
        elif cls == "STAR":
            u, g = rng.normal(17.5, 2.0), rng.normal(17.0, 1.9)
            r, i, z = rng.normal(16.8, 1.8), rng.normal(16.5, 1.8), rng.normal(16.3, 1.8)
            redshift = rng.normal(0.0, 0.0002)
        else:  # QSO
            u, g = rng.normal(20.0, 1.2), rng.normal(19.5, 1.1)
            r, i, z = rng.normal(19.0, 1.1), rng.normal(18.7, 1.1), rng.normal(18.5, 1.1)
            redshift = rng.exponential(1.2)

        rows.append({
            "alpha": rng.uniform(0, 360),
            "delta": rng.uniform(-90, 90),
            "u": u, "g": g, "r": r, "i": i, "z": z,
            "run_ID": int(rng.integers(1, 1000)),
            "cam_col": int(rng.integers(1, 7)),
            "field_ID": int(rng.integers(1, 500)),
            "redshift": max(redshift, 0.0),
            "plate": int(rng.integers(1, 8000)),
            "MJD": int(rng.integers(51000, 59000)),
            "fiber_ID": int(rng.integers(1, 1001)),
            "class": cls,
        })

    df = pd.DataFrame(rows)
    # Inject ~1 % missing values to demonstrate imputation
    for col in ["u", "g", "r", "i", "z"]:
        mask = rng.random(n) < 0.01
        df.loc[mask, col] = np.nan

    # Save next to the script if the Downloads directory does not exist
    save_path = csv_path if os.path.isdir(os.path.dirname(csv_path)) else os.path.basename(csv_path)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Synthetic dataset saved to '{save_path}'.")
    return df


# ─────────────────────────────────────────────
# 2. Preprocess the data
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("\n=== 2. Preprocessing ===")

    # --- a) Missing-data technique ---
    # Use median imputation for photometric magnitudes
    mag_cols = ["u", "g", "r", "i", "z"]
    missing_before = df[mag_cols].isna().sum().sum()
    imputer = SimpleImputer(strategy="median")
    df[mag_cols] = imputer.fit_transform(df[mag_cols])
    missing_after = df[mag_cols].isna().sum().sum()
    print(f"  Missing values in photometric bands: {missing_before} -> {missing_after} (median imputation)")

    # --- b) Feature engineering ---
    # FE-1: color indices (magnitude differences) – standard in stellar astronomy
    df["u_g"] = df["u"] - df["g"]   # ultra-violet to green color
    df["g_r"] = df["g"] - df["r"]   # green to red color
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    print("  Feature engineering 1: color indices u_g, g_r, r_i, i_z added")

    # FE-2: log-transformed redshift (compresses the long tail)
    df["log_redshift"] = np.log1p(df["redshift"])
    print("  Feature engineering 2: log1p(redshift) added")

    # --- c) Feature selection with ANOVA F-test (SelectKBest) ---
    feature_cols = [
        "alpha", "delta", "u", "g", "r", "i", "z",
        "run_ID", "cam_col", "field_ID",
        "redshift", "plate", "MJD", "fiber_ID",
        "u_g", "g_r", "r_i", "i_z", "log_redshift",
    ]
    le = LabelEncoder()
    y = le.fit_transform(df["class"])

    selector = SelectKBest(score_func=f_classif, k=10)
    selector.fit(df[feature_cols], y)
    support = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in support]
    print(f"  Feature selection (SelectKBest, k=10): {selected_features}")

    return df, selected_features, le, y


# ─────────────────────────────────────────────
# 3 & 4. Subset + train/test split
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame, selected_features: list, y):
    print("\n=== 3. Subsetting & 4. Train/Test Split ===")
    X = df[selected_features]
    # Standardise features (important for Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"  Training samples : {X_train.shape[0]}  |  Test samples : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, X_scaled


# ─────────────────────────────────────────────
# 5 & 6. Build and run models
# ─────────────────────────────────────────────
def train_and_predict(X_train, X_test, y_train):
    print("\n=== 5 & 6. Training & Prediction ===")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("  Logistic Regression trained.")

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print("  Decision Tree trained.")

    return lr, dt, y_pred_lr, y_pred_dt


# ─────────────────────────────────────────────
# 7, 8 & 9. Display results, confusion matrices,
#           precision / recall / F-measure
# ─────────────────────────────────────────────
def evaluate(y_test, y_pred, model_name: str, class_names: list):
    print(f"\n=== {model_name} – Results ===")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {fname}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_class_distribution(df: pd.DataFrame):
    """Bar chart of the target class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["class"].value_counts()
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title("Class Distribution in SDSS-17 Dataset")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for i, (cls, cnt) in enumerate(counts.items()):
        ax.text(i, cnt + 50, f"{cnt:,}", ha="center", fontsize=10)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Class distribution plot saved -> {path}")


def plot_feature_importance(dt, feature_names: list):
    """Bar chart of Decision Tree feature importances."""
    importances = pd.Series(dt.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Decision Tree – Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/feature_importance_dt.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Feature importance plot saved -> {path}")


def plot_metrics_comparison(lr_metrics: dict, dt_metrics: dict):
    """Side-by-side bar chart comparing model metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [lr_metrics[m] for m in metrics], width, label="Logistic Regression", color="#4C72B0")
    ax.bar(x + width / 2, [dt_metrics[m] for m in metrics], width, label="Decision Tree", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    for i, m in enumerate(metrics):
        ax.text(i - width / 2, lr_metrics[m] + 0.01, f"{lr_metrics[m]:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, dt_metrics[m] + 0.01, f"{dt_metrics[m]:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/metrics_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Metrics comparison plot saved -> {path}")


def plot_colour_scatter(df: pd.DataFrame):
    """Scatter plot of two colour indices coloured by class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"GALAXY": "#4C72B0", "STAR": "#DD8452", "QSO": "#55A868"}
    for cls, grp in df.groupby("class"):
        ax.scatter(grp["u_g"], grp["g_r"], s=4, alpha=0.4, label=cls, color=palette.get(cls, "grey"))
    ax.set_xlabel("u − g (color index)")
    ax.set_ylabel("g − r (color index)")
    ax.set_title("Color-Color Diagram by Class")
    ax.legend(markerscale=3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/color_color_diagram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Color-color diagram saved -> {path}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    csv_path = os.path.join(
        os.path.expanduser("~"), "Downloads", "StellarClassificationDatasetSDSS17.csv"
    )

    df = load_or_generate_dataset(csv_path)
    print(f"\nDataset shape: {df.shape}")
    print(df["class"].value_counts().to_string())

    plot_class_distribution(df)

    df, selected_features, le, y = preprocess(df)

    # Draw scatter after feature engineering (u_g / g_r now available)
    plot_colour_scatter(df)

    X_train, X_test, y_train, y_test, _ = split_data(df, selected_features, y)
    class_names = list(le.classes_)

    lr, dt, y_pred_lr, y_pred_dt = train_and_predict(X_train, X_test, y_train)

    lr_metrics = evaluate(y_test, y_pred_lr, "Logistic Regression", class_names)
    dt_metrics = evaluate(y_test, y_pred_dt, "Decision Tree", class_names)

    plot_feature_importance(dt, selected_features)
    plot_metrics_comparison(lr_metrics, dt_metrics)

    print("\n=== Done – all outputs saved to ./outputs/ ===")


if __name__ == "__main__":
    main()
