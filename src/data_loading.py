"""
Data loading and preprocessing utilities for the Global Shark Attack dataset.
"""

import os
import numpy as np
import pandas as pd


def load_shark_data(shark_data_path="data/global-shark-attack.csv", location_coords_path="data/location-coordinates.csv"):
    """Load and preprocess the shark attack dataset.
    Pull year and month from the shark attack dataset, longitude and latitude from the location coordinates dataset, and combine them."""
    if not os.path.exists(shark_data_path):
        raise FileNotFoundError(f"Shark data CSV file not found at {shark_data_path}")
    if not os.path.exists(location_coords_path):
        raise FileNotFoundError(f"Location coordinates CSV file not found at {location_coords_path}")

    # Read shark attack data
    shark_df = pd.read_csv(shark_data_path, sep=";", engine="python", dtype=str)
    shark_df.columns = shark_df.columns.str.strip()

    # Read location coordinates data
    coords_df = pd.read_csv(location_coords_path, sep=",", engine="python", dtype=str)
    coords_df.columns = coords_df.columns.str.strip()

    # Ensure both datasets have the same number of rows
    if len(shark_df) != len(coords_df):
        raise ValueError("The shark data and location coordinates files must have the same number of rows.")

    # Extract year and month from shark data
    shark_df["Year"] = pd.to_numeric(shark_df["Year"], errors="coerce")
    shark_df["Month"] = pd.to_datetime(shark_df["Date"], errors="coerce").dt.month

    # Extract longitude and latitude from location coordinates
    coords_df["Longitude"] = pd.to_numeric(coords_df["lon"], errors="coerce")
    coords_df["Latitude"] = pd.to_numeric(coords_df["lat"], errors="coerce")

    # Combine the datasets
    combined_df = pd.DataFrame({
        "Year": shark_df["Year"],
        "Month": shark_df["Month"],
        "Longitude": coords_df["Longitude"],
        "Latitude": coords_df["Latitude"]
    })

    # Drop rows with missing values
    combined_df = combined_df.dropna()

    # Convert to numpy array
    result_array = combined_df.values

    return result_array

def make_negative_dataset(PositiveData):
    negativeData = []
    for i in range(len(PositiveData) * 9):
        ranYear = np.random.randint(2010, 2022)
        ranMonth = np.random.randint(1, 13)
        ranLong = np.random.uniform(-180, 180)
        ranLat = np.random.uniform(-90, 90)

        # Round to a random number of decimal places between 4 and 10
        decimal_places = np.random.randint(4, 11)
        ranLong = round(ranLong, decimal_places)
        ranLat = round(ranLat, decimal_places)

        dataPoint = [ranYear, ranMonth, ranLong, ranLat]
        negativeData.append(dataPoint)
    return np.array(negativeData)
        



if __name__ == "__main__":
    # Load positive data
    print("Loading positive shark attack data...")
    positive_data = load_shark_data()
    positive_data = positive_data[(positive_data[:, 0] >= 2010) & (positive_data[:, 0] <= 2022)]
    print(f"Loaded positive dataset: {positive_data.shape}")
    print(f"Sample rows:\n{positive_data[:5]}")
    
    # Generate negative data
    print("\nGenerating negative dataset...")
    negative_data = make_negative_dataset(positive_data)
    print(f"Generated negative dataset: {negative_data.shape}")
    print(f"Sample rows:\n{negative_data[:5]}")
    
    # Create output directory if it doesn't exist
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrames for easier CSV saving
    positive_df = pd.DataFrame(positive_data, columns=['Year', 'Month', 'Longitude', 'Latitude'])
    negative_df = pd.DataFrame(negative_data, columns=['Year', 'Month', 'Longitude', 'Latitude'])
    
    # Save to CSV files
    positive_path = os.path.join(output_dir, "positive_shark_attacks.csv")
    negative_path = os.path.join(output_dir, "negative_samples.csv")
    
    positive_df.to_csv(positive_path, index=False)
    negative_df.to_csv(negative_path, index=False)
    
    print(f"\n✓ Saved positive data to: {positive_path}")
    print(f"✓ Saved negative data to: {negative_path}")
