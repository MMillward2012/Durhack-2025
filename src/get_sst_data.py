"""
Simple SST Data Fetcher for Shark Attack Data
Gets SST values for specific dates/locations and appends to CSV files.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xarray as xr
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')


def get_sst_value(lat, lon, year, month):
    """
    Get SST value for a specific location and date.
    
    Args:
        lat: Latitude
        lon: Longitude  
        year: Year (int)
        month: Month (int)
        
    Returns:
        SST value in Celsius, or None if failed
    """
    try:
        # Handle longitude wrapping (NOAA uses 0-360)
        if lon < 0:
            lon = lon + 360
            
        # Create date string (use 1st of month)
        date_str = f"{year}-{month:02d}-01"
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # NOAA OPeNDAP URL
        opendap_url = "https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR"
        file_url = f"{opendap_url}/{year}{month:02d}/oisst-avhrr-v02r01.{date.strftime('%Y%m%d')}.nc"
        
        print(f"  Getting SST at ({lat:.3f}, {lon:.3f}) for {date_str}...")
        
        # Open remote file and get SST at location
        ds = xr.open_dataset(file_url, engine='netcdf4')
        
        # Get SST value at location (with small buffer for interpolation)
        sst_value = ds['sst'].sel(lat=lat, lon=lon, method='nearest').values.item()
        
        ds.close()
        
        # Check if valid
        if np.isnan(sst_value) or sst_value < -5 or sst_value > 50:
            print(f"    ✗ Invalid SST: {sst_value}")
            return None
            
        print(f"    ✓ SST: {sst_value:.2f}°C")
        return float(sst_value)
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def process_shark_data():
    """
    Main function: Load shark data, get SST values, save updated CSVs.
    """
    print("🌊 SIMPLE SST FETCHER FOR SHARK ATTACKS")
    print("="*60)
    
    # File paths
    data_dir = Path('data/processed')
    positive_file = data_dir / 'positive_shark_attacks.csv'
    negative_file = data_dir / 'negative_samples.csv'
    
    # Load data
    print("Loading shark attack data...")
    positive_df = pd.read_csv(positive_file)
    negative_df = pd.read_csv(negative_file)
    
    print(f"✓ Loaded {len(positive_df)} positive samples")
    print(f"✓ Loaded {len(negative_df)} negative samples")
    
    # Process positive samples
    print(f"\n{'='*60}")
    print("PROCESSING POSITIVE SHARK ATTACKS")
    print(f"{'='*60}")
    
    positive_df['SST_Celsius'] = None
    successful = 0
    
    for i, row in positive_df.iterrows():
        print(f"\n[{i+1}/{len(positive_df)}] Record {i+1}")
        
        sst = get_sst_value(row['Latitude'], row['Longitude'], 
                           int(row['Year']), int(row['Month']))
        
        if sst is not None:
            positive_df.at[i, 'SST_Celsius'] = sst
            successful += 1
        
        # Progress update every 1000 records
        if (i + 1) % 1000 == 0:
            print(f"\n  Progress: {i+1}/{len(positive_df)} - Success rate: {(successful/(i+1)*100):.1f}%")
    
    print(f"\n✓ Positive samples: {successful}/{len(positive_df)} successful")
    
    # Process negative samples  
    print(f"\n{'='*60}")
    print("PROCESSING NEGATIVE SAMPLES")
    print(f"{'='*60}")
    
    negative_df['SST_Celsius'] = None
    successful = 0
    
    for i, row in negative_df.iterrows():
        print(f"\n[{i+1}/{len(negative_df)}] Record {i+1}")
        
        sst = get_sst_value(row['Latitude'], row['Longitude'], 
                           int(row['Year']), int(row['Month']))
        
        if sst is not None:
            negative_df.at[i, 'SST_Celsius'] = sst
            successful += 1

        # Progress update every 1000 records
        if (i + 1) % 1000 == 0:
            print(f"\n  Progress: {i+1}/{len(negative_df)} - Success rate: {(successful/(i+1)*100):.1f}%")
    
    print(f"\n✓ Negative samples: {successful}/{len(negative_df)} successful")
    
    # Save updated files
    print(f"\n{'='*60}")
    print("SAVING UPDATED FILES")
    print(f"{'='*60}")
    
    positive_output = data_dir / 'positive_shark_attacks_with_sst.csv'
    negative_output = data_dir / 'negative_samples_with_sst.csv'
    
    positive_df.to_csv(positive_output, index=False)
    negative_df.to_csv(negative_output, index=False)
    
    print(f"✓ Saved: {positive_output}")
    print(f"✓ Saved: {negative_output}")
    
    # Show final stats
    pos_sst = positive_df['SST_Celsius'].dropna()
    neg_sst = negative_df['SST_Celsius'].dropna()
    
    if len(pos_sst) > 0:
        print(f"\nPositive SST stats: {pos_sst.mean():.1f}°C ± {pos_sst.std():.1f}°C")
    if len(neg_sst) > 0:
        print(f"Negative SST stats: {neg_sst.mean():.1f}°C ± {neg_sst.std():.1f}°C")
    
    print(f"\n🎉 DONE! Your CSV files now have SST_Celsius column.")


def main():
    """Run with small test first."""
    
    process_shark_data()


if __name__ == "__main__":
    main()