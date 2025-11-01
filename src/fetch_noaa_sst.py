"""
NOAA OISST Data Fetcher for Shark Attack Heatmap
Fetches Sea Surface Temperature data for coastal regions and converts to CSV/grid format.

NOAA OISST (Optimum Interpolation Sea Surface Temperature) provides global SST data.
Data source: https://www.ncei.noaa.gov/products/optimum-interpolation-sst
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path


class NOAAOISSTFetcher:
    """
    Fetch NOAA OISST data for coastal regions.
    """
    
    # High-risk coastal regions for shark attacks
    COASTAL_REGIONS = {
        'Australia_East': {'lat_min': -38, 'lat_max': -16, 'lon_min': 150, 'lon_max': 155},
        'Australia_West': {'lat_min': -35, 'lat_max': -20, 'lon_min': 113, 'lon_max': 118},
        'USA_Florida': {'lat_min': 24, 'lat_max': 31, 'lon_min': -87, 'lon_max': -80},
        'USA_California': {'lat_min': 32, 'lat_max': 42, 'lon_min': -125, 'lon_max': -117},
        'South_Africa': {'lat_min': -35, 'lat_max': -28, 'lon_min': 16, 'lon_max': 33},
        'Hawaii': {'lat_min': 18, 'lat_max': 23, 'lon_min': -161, 'lon_max': -154},
        'Brazil': {'lat_min': -30, 'lat_max': -3, 'lon_min': -50, 'lon_max': -34},
        'Mexico_Pacific': {'lat_min': 16, 'lat_max': 32, 'lon_min': -117, 'lon_max': -105},
    }
    
    # NOAA ERDDAP server for OISST data
    ERDDAP_BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv"
    
    def __init__(self, output_dir='data/raw'):
        """
        Initialize the OISST fetcher.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_sst_data(self, region_name, start_date=None, end_date=None, grid_resolution=0.5):
        """
        Fetch SST data for a specific coastal region.
        
        Args:
            region_name: Name of region from COASTAL_REGIONS
            start_date: Start date (YYYY-MM-DD), defaults to 1 year ago
            end_date: End date (YYYY-MM-DD), defaults to today
            grid_resolution: Grid cell size in degrees (default 0.5)
            
        Returns:
            pandas.DataFrame with SST data
        """
        if region_name not in self.COASTAL_REGIONS:
            raise ValueError(f"Region {region_name} not found. Available: {list(self.COASTAL_REGIONS.keys())}")
        
        region = self.COASTAL_REGIONS[region_name]
        
        # Set default dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        print(f"Fetching SST data for {region_name}...")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Region: lat {region['lat_min']}-{region['lat_max']}, lon {region['lon_min']}-{region['lon_max']}")
        
        # Build ERDDAP query URL
        params = {
            'time': f'{start_date}T00:00:00Z',
            'time_end': f'{end_date}T00:00:00Z',
            'latitude': f'{region["lat_min"]}:{region["lat_max"]}',
            'longitude': f'{region["lon_min"]}:{region["lon_max"]}',
        }
        
        # Construct URL (simplified - you may need to adjust based on ERDDAP server)
        # Note: This is a template. Real ERDDAP URLs vary by server.
        url = f"{self.ERDDAP_BASE_URL}?sst[({params['time']}):1:({params['time_end']})][({params['latitude']})][({params['longitude']})]"
        
        try:
            print(f"Fetching from ERDDAP server...")
            print(f"URL: {url[:100]}...")
            
            # Note: This may take a while for large regions/time ranges
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), skiprows=[1])  # Skip units row
            
            print(f"✓ Fetched {len(df)} data points")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching data: {e}")
            print("\nNote: NOAA ERDDAP servers have usage limits.")
            print("Alternative: Download data manually from https://www.ncei.noaa.gov/products/optimum-interpolation-sst")
            return None
    
    def create_grid_from_sst(self, sst_df, grid_resolution=0.5, preserve_time=True):
        """
        Convert SST data to a regular grid for heatmap.
        
        Args:
            sst_df: DataFrame with SST data (columns: time, latitude, longitude, sst)
            grid_resolution: Grid cell size in degrees
            preserve_time: If True, keep date information (one row per date per cell)
                          If False, average across all dates (one row per cell)
            
        Returns:
            DataFrame with gridded SST values
        """
        if sst_df is None or sst_df.empty:
            return pd.DataFrame()
        
        print(f"Creating grid with {grid_resolution}° resolution...")
        
        # Parse time column if it exists and convert to date
        if 'time' in sst_df.columns:
            sst_df['date'] = pd.to_datetime(sst_df['time']).dt.date
            sst_df['year'] = pd.to_datetime(sst_df['time']).dt.year
            sst_df['month'] = pd.to_datetime(sst_df['time']).dt.month
            sst_df['day'] = pd.to_datetime(sst_df['time']).dt.day
        
        # Group by grid cells
        sst_df['lat_grid'] = (sst_df['latitude'] / grid_resolution).round() * grid_resolution
        sst_df['lon_grid'] = (sst_df['longitude'] / grid_resolution).round() * grid_resolution
        
        # Group by location and optionally time
        if preserve_time and 'date' in sst_df.columns:
            group_cols = ['lat_grid', 'lon_grid', 'date', 'year', 'month', 'day']
            grid_df = sst_df.groupby(group_cols).agg({
                'sst': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()
            
            # Flatten column names
            grid_df.columns = ['latitude', 'longitude', 'date', 'year', 'month', 'day', 
                              'sst_mean', 'sst_std', 'sst_min', 'sst_max', 'data_points']
            print(f"✓ Created grid with {len(grid_df)} cells (with dates)")
        else:
            # Average across all time periods
            grid_df = sst_df.groupby(['lat_grid', 'lon_grid']).agg({
                'sst': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()
            
            # Flatten column names
            grid_df.columns = ['latitude', 'longitude', 'sst_mean', 'sst_std', 'sst_min', 'sst_max', 'data_points']
            print(f"✓ Created grid with {len(grid_df)} cells (averaged over time)")
        
        return grid_df
    
    def fetch_all_regions(self, start_date=None, end_date=None, save_both=True):
        """
        Fetch SST data for all defined coastal regions.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_both: If True, save both time-preserved and time-averaged versions
            
        Returns:
            Dictionary of DataFrames, one per region
        """
        all_data = {}
        
        for region_name in self.COASTAL_REGIONS.keys():
            print(f"\n{'='*60}")
            sst_df = self.fetch_sst_data(region_name, start_date, end_date)
            
            if sst_df is not None:
                if save_both:
                    # Save version WITH dates (for matching with shark attacks)
                    grid_df_with_time = self.create_grid_from_sst(sst_df, preserve_time=True)
                    output_file_time = self.output_dir / f'sst_grid_{region_name.lower()}_with_dates.csv'
                    grid_df_with_time.to_csv(output_file_time, index=False)
                    print(f"✓ Saved (with dates) to {output_file_time}")
                    
                    # Save version WITHOUT dates (for general heatmap)
                    grid_df_avg = self.create_grid_from_sst(sst_df, preserve_time=False)
                    output_file_avg = self.output_dir / f'sst_grid_{region_name.lower()}_averaged.csv'
                    grid_df_avg.to_csv(output_file_avg, index=False)
                    print(f"✓ Saved (averaged) to {output_file_avg}")
                    
                    all_data[region_name] = {
                        'with_time': grid_df_with_time,
                        'averaged': grid_df_avg
                    }
                else:
                    # Default: save with time preserved
                    grid_df = self.create_grid_from_sst(sst_df, preserve_time=True)
                    all_data[region_name] = grid_df
                    
                    output_file = self.output_dir / f'sst_grid_{region_name.lower()}.csv'
                    grid_df.to_csv(output_file, index=False)
                    print(f"✓ Saved to {output_file}")
            else:
                print(f"✗ Failed to fetch data for {region_name}")
        
        return all_data
    
    def create_sample_data(self, with_dates=True):
        """
        Create sample SST data for testing (when API is unavailable).
        
        Args:
            with_dates: If True, include date information for past year
        
        Returns:
            DataFrame with sample SST data
        """
        print("Creating sample SST data for testing...")
        
        sample_data = []
        
        # Generate dates for past year if requested
        if with_dates:
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=365),
                end=datetime.now(),
                freq='7D'  # Weekly samples
            )
        else:
            dates = [None]
        
        for date in dates:
            for region_name, bounds in self.COASTAL_REGIONS.items():
                # Create grid
                lats = np.arange(bounds['lat_min'], bounds['lat_max'], 0.5)
                lons = np.arange(bounds['lon_min'], bounds['lon_max'], 0.5)
                
                for lat in lats:
                    for lon in lons:
                        # Simulate SST (warmer near equator, seasonal variation)
                        base_temp = 28 - abs(lat) * 0.4  # Cooler away from equator
                        
                        # Add seasonal variation based on month
                        if date is not None:
                            month = date.month
                            # Northern hemisphere: warmer in Jun-Aug, cooler in Dec-Feb
                            if lat > 0:
                                seasonal_effect = 3 * np.cos((month - 7) * np.pi / 6)
                            # Southern hemisphere: opposite
                            else:
                                seasonal_effect = 3 * np.cos((month - 1) * np.pi / 6)
                        else:
                            seasonal_effect = 0
                        
                        random_var = np.random.normal(0, 1.5)
                        sst = base_temp + seasonal_effect + random_var
                        
                        data_point = {
                            'region': region_name,
                            'latitude': lat,
                            'longitude': lon,
                            'sst_mean': sst,
                            'sst_std': np.random.uniform(0.5, 2.0),
                            'sst_min': sst - 2,
                            'sst_max': sst + 2,
                            'data_points': 30
                        }
                        
                        if date is not None:
                            data_point['date'] = date.date()
                            data_point['year'] = date.year
                            data_point['month'] = date.month
                            data_point['day'] = date.day
                        
                        sample_data.append(data_point)
        
        df = pd.DataFrame(sample_data)
        
        # Save combined sample data
        if with_dates:
            output_file = self.output_dir / 'sst_grid_sample_with_dates.csv'
            print(f"✓ Sample data with dates saved to {output_file}")
        else:
            output_file = self.output_dir / 'sst_grid_sample.csv'
            print(f"✓ Sample data (no dates) saved to {output_file}")
        
        df.to_csv(output_file, index=False)
        
        return df


def main():
    """
    Example usage of NOAAOISSTFetcher.
    """
    print("🌊 NOAA OISST Data Fetcher for Shark Attack Heatmap")
    print("=" * 60)
    
    fetcher = NOAAOISSTFetcher()
    
    # Option 1: Try to fetch real data (may fail due to API limits/connectivity)
    print("\nAttempting to fetch real NOAA data...")
    print("Note: This may take several minutes and requires internet connection.")
    print("If this fails, sample data will be created instead.\n")
    
    try:
        # Fetch last 90 days of data for Florida (smaller test)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        sst_df = fetcher.fetch_sst_data('USA_Florida', start_date, end_date)
        
        if sst_df is not None and not sst_df.empty:
            grid_df = fetcher.create_grid_from_sst(sst_df)
            output_file = fetcher.output_dir / 'sst_grid_usa_florida.csv'
            grid_df.to_csv(output_file, index=False)
            print(f"\n✓ Success! SST data saved to {output_file}")
        else:
            raise Exception("No data returned")
            
    except Exception as e:
        print(f"\n⚠ Could not fetch real data: {e}")
        print("\n📊 Creating sample data instead...")
        fetcher.create_sample_data(with_dates=True)
    
    print("\n" + "=" * 60)
    print("✓ Data fetching complete!")
    print("\nNext steps:")
    print("  1. Check data/raw/ for SST CSV files")
    print("  2. Use this data in your shark attack model")
    print("  3. Merge with GSAF attack data by location AND date")
    print("\n📊 Data format:")
    print("  • *_with_dates.csv: Has date/year/month/day columns for temporal matching")
    print("  • *_averaged.csv: Averaged over time for general heatmap")
    print("\n💡 Integration example:")
    print("  # Match attacks to SST by location AND date")
    print("  merged = attacks_df.merge(")
    print("      sst_df,")
    print("      left_on=['lat_grid', 'lon_grid', 'attack_date'],")
    print("      right_on=['latitude', 'longitude', 'date'],")
    print("      how='left'")
    print("  )")
    print("\nAlternative data sources:")
    print("  - https://www.ncei.noaa.gov/products/optimum-interpolation-sst")
    print("  - https://coastwatch.pfeg.noaa.gov/erddap/griddap/")
    print("  - Manual download from NOAA website")


if __name__ == "__main__":
    main()
