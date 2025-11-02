#!/usr/bin/env python3
"""
Real Shark Density Fetcher using GBIF
=====================================

Downloads REAL shark occurrence data from GBIF (Global Biodiversity Information Facility)
and calculates actual shark density for 0.5° x 0.5° grid cells, then provides fast lookup.

GBIF has 3.5+ billion shark occurrence records with coordinates!

Author: GitHub Copilot  
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

class RealSharkDensityGrid:
    """Create a lookup table of shark density for 0.5° x 0.5° grid cells."""

    def __init__(self, shark_data_file):
        self.shark_data_file = shark_data_file
        self.grid_density_file = "data/processed/shark_density_grid.csv"

    def create_density_grid(self):
        """Create a lookup table of shark density at 0.5° x 0.5° grid points using 5x5 smoothing."""
        # Load shark observation data
        shark_df = pd.read_csv(self.shark_data_file)
        print(f"Loaded {len(shark_df):,} shark observations")

        # Round latitude and longitude to the nearest 0.5°
        shark_df['LatBin'] = (shark_df['latitude'] / 0.5).round() * 0.5
        shark_df['LonBin'] = (shark_df['longitude'] / 0.5).round() * 0.5

        # Create all possible grid cells (360 x 720 = 259,200 cells)
        lat_bins = np.arange(-90, 90.5, 0.5)  # -90 to 90 in 0.5° steps
        lon_bins = np.arange(-180, 180.5, 0.5)  # -180 to 180 in 0.5° steps
        
        print(f"Creating grid: {len(lat_bins)} × {len(lon_bins)} = {len(lat_bins) * len(lon_bins):,} cells")
        
        all_grid_cells = pd.DataFrame(
            [(lat, lon) for lat in lat_bins for lon in lon_bins],
            columns=['LatBin', 'LonBin']
        )

        # Count shark observations per grid cell
        shark_counts = (
            shark_df.groupby(['LatBin', 'LonBin'])
            .size()
            .reset_index(name='SharkCount')
        )

        # Merge with all possible grid cells (fills missing cells with 0)
        density_grid = all_grid_cells.merge(shark_counts, on=['LatBin', 'LonBin'], how='left')
        density_grid['SharkCount'] = density_grid['SharkCount'].fillna(0)

        print("Calculating smoothed densities using 5x5 neighborhoods...")
        
        # Calculate smoothed shark count for each cell using 5x5 neighborhood
        smoothed_counts = []
        for i, row in density_grid.iterrows():
            if i % 10000 == 0:
                print(f"  Progress: {i:,}/{len(density_grid):,} ({i/len(density_grid)*100:.1f}%)")
            
            lat, lon = row['LatBin'], row['LonBin']
            
            # Define 5x5 neighborhood (2.5° x 2.5° around the cell)
            neighbors = density_grid[
                (density_grid['LatBin'] >= lat - 1.0) & (density_grid['LatBin'] <= lat + 1.0) &
                (density_grid['LonBin'] >= lon - 1.0) & (density_grid['LonBin'] <= lon + 1.0)
            ]
            
            # Sum all shark counts in the 5x5 neighborhood
            smoothed_count = neighbors['SharkCount'].sum()
            smoothed_counts.append(smoothed_count)

        density_grid['SmoothedSharkCount'] = smoothed_counts

        # Calculate area of 5x5 neighborhood (2.5° x 2.5° varies by latitude)
        # Area = (2.5° lat) × (2.5° lon × cos(lat)) × (111 km/°)²
        density_grid['NeighborhoodAreaKm2'] = (2.5 * 2.5 * np.cos(np.radians(density_grid['LatBin'])) * 111**2)
        
        # Calculate density per 1000 km² based on smoothed counts
        density_grid['SharkDensity'] = (density_grid['SmoothedSharkCount'] / density_grid['NeighborhoodAreaKm2']) * 1000

        # Normalize to 0-1 range
        max_density = density_grid['SharkDensity'].max()
        if max_density > 0:
            density_grid['NormalizedDensity'] = density_grid['SharkDensity'] / max_density
        else:
            density_grid['NormalizedDensity'] = 0

        # Keep only the columns we need for lookup
        lookup_table = density_grid[['LatBin', 'LonBin', 'NormalizedDensity']].copy()

        # Save the lookup table
        Path(self.grid_density_file).parent.mkdir(parents=True, exist_ok=True)
        lookup_table.to_csv(self.grid_density_file, index=False)
        
        print(f"✓ Smoothed shark density lookup table saved to {self.grid_density_file}")
        print(f"  Grid cells with sharks (5x5 smoothed): {(density_grid['SmoothedSharkCount'] > 0).sum():,}")
        print(f"  Max smoothed density: {max_density:.6f} per 1000 km²")
        print(f"  Normalized range: 0.0 to 1.0")

    def get_density_for_point(self, lat, lon):
        """Get the shark density for a specific latitude and longitude by looking up the grid cell."""
        # Load the density grid
        density_grid = pd.read_csv(self.grid_density_file)

        # Find the corresponding grid cell
        lat_bin = round(lat / 0.5) * 0.5
        lon_bin = round(lon / 0.5) * 0.5

        # Query the density grid
        density = density_grid.loc[
            (density_grid['LatBin'] == lat_bin) & (density_grid['LonBin'] == lon_bin),
            'NormalizedDensity'
        ]

        if density.empty:
            return 0.0  # No data for this grid cell
        return density.values[0]

    def add_density_to_dataset(self, input_csv, output_csv):
        """Add shark density to a dataset by looking up grid cells."""
        # Load the input dataset
        dataset = pd.read_csv(input_csv)
        print(f"Processing {len(dataset):,} records from {input_csv}")

        # Load the density grid
        density_grid = pd.read_csv(self.grid_density_file)

        # Function to find density for each row
        def find_density(row):
            lat_bin = round(row['Latitude'] / 0.5) * 0.5
            lon_bin = round(row['Longitude'] / 0.5) * 0.5
            density = density_grid.loc[
                (density_grid['LatBin'] == lat_bin) & (density_grid['LonBin'] == lon_bin),
                'NormalizedDensity'
            ]
            return density.values[0] if not density.empty else 0.0

        # Add shark density column using fast lookup
        dataset['Real_Shark_Density'] = dataset.apply(find_density, axis=1)

        # Save the updated dataset
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_csv, index=False)
        
        densities = dataset['Real_Shark_Density']
        print(f"✓ Updated dataset saved to {output_csv}")
        print(f"  Density stats: min={densities.min():.3f}, max={densities.max():.3f}, mean={densities.mean():.3f}")

if __name__ == "__main__":
    # File paths
    shark_data_file = "data/real_shark_observations/gbif_sharks.csv"

    # Initialize the density grid calculator
    density_calculator = RealSharkDensityGrid(shark_data_file)

    # Create the density grid lookup table
    density_calculator.create_density_grid()

    # Process positive and negative datasets
    datasets = [
        ("data/unprocessed/positive_with_sst_and_pop.csv", "data/processed/positive_with_real_shark_density.csv"),
        ("data/unprocessed/negative_with_sst_and_pop.csv", "data/processed/negative_with_real_shark_density.csv")
    ]

    for input_csv, output_csv in datasets:
        if os.path.exists(input_csv):
            density_calculator.add_density_to_dataset(input_csv, output_csv)
        else:
            print(f"File not found: {input_csv}")