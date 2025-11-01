"""
Data loading and preprocessing utilities for the Global Shark Attack dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_shark_data(path: str = "data/global-shark-attack.csv"):
    """Load and preprocess the shark attack dataset.
    Returns a numpy array with cleaned features."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at {path}")

    # Read semicolon-delimited file
    df = pd.read_csv(path, sep=";", engine="python", dtype=str)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Select useful features
    features = ["Year", "Age", "Sex", "Country", "Activity", "Type", "Species"]
    Xdf = df[features].copy()

    # Clean numeric fields
    Xdf["Year"] = pd.to_numeric(Xdf["Year"], errors="coerce")
    Xdf["Age"] = pd.to_numeric(Xdf["Age"], errors="coerce")

    # Clean categorical fields
    Xdf["Sex"] = Xdf["Sex"].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    for cat in ["Country", "Activity", "Type", "Species"]:
        Xdf[cat] = Xdf[cat].astype(str).str.strip().fillna("Unknown")

    # Group rare species into "Other"
    top_species = Xdf["Species"].value_counts().nlargest(15).index.tolist()
    Xdf["Species"] = Xdf["Species"].where(Xdf["Species"].isin(top_species), other="Other")

    # Fill missing numeric values with median
    Xdf["Year"] = Xdf["Year"].fillna(Xdf["Year"].median())
    Xdf["Age"] = Xdf["Age"].fillna(Xdf["Age"].median())

    # One-hot encode categoricals
    categoricals = ["Sex", "Country", "Activity", "Type", "Species"]
    X_cat = pd.get_dummies(Xdf[categoricals], dummy_na=False)

    # Scale numeric features
    X_num = Xdf[["Year", "Age"]].astype(float)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Combine numeric and categorical features
    X = np.hstack([X_num_scaled, X_cat.values]).astype(np.float32)

    return X


if __name__ == "__main__":
    # Quick test
    try:
        X = load_shark_data()
        print("Loaded shark dataset:")
        print("Shape:", X.shape)
    except Exception as e:
        print("Error loading data:", e)
