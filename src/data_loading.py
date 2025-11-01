"""
Data loading and preprocessing utilities for the Global Shark Attack dataset.
"""

import os
import numpy as np
import pandas as pd


def load_shark_data(path: str = "data/global-shark-attack.csv"):
    """Load and preprocess the shark attack dataset.
    Returns a numpy array with Date, Longitude, and Latitude."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at {path}")

    # Read semicolon-delimited file
    df = pd.read_csv(path, sep=";", engine="python", dtype=str)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Select specific columns
    columns = ["Date", "Longitude", "Latitude"]
    Xdf = df[columns].copy()

    # Fill missing values with "Unknown" for Date and 0 for coordinates
    Xdf["Date"] = Xdf["Date"].fillna("Unknown")
    Xdf["Longitude"] = pd.to_numeric(Xdf["Longitude"], errors="coerce").fillna(0)
    Xdf["Latitude"] = pd.to_numeric(Xdf["Latitude"], errors="coerce").fillna(0)

    # Convert to numpy array
    X = Xdf.values

    return X


if __name__ == "__main__":
    # Quick test
    try:
        X = load_shark_data()
        print("Loaded shark dataset:")
        print("Shape:", X.shape)
        print("Sample rows:", X[:5])
    except Exception as e:
        print("Error loading data:", e)
