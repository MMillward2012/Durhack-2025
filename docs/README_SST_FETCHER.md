# NOAA SST Data Fetcher

This script fetches Sea Surface Temperature (SST) data from NOAA for coastal regions relevant to shark attack prediction.

## Usage

### Basic Usage
```bash
python src/fetch_noaa_sst.py
```

This will:
1. Attempt to fetch real NOAA OISST data
2. If that fails (due to API limits or connectivity), create sample data
3. Save gridded SST data to `data/raw/sst_grid_*.csv`

### Using in Your Code
```python
from fetch_noaa_sst import NOAAOISSTFetcher

# Initialize fetcher
fetcher = NOAAOISSTFetcher(output_dir='data/raw')

# Fetch data for a specific region
sst_df = fetcher.fetch_sst_data('USA_Florida', start_date='2024-01-01', end_date='2024-12-31')

# Convert to grid
grid_df = fetcher.create_grid_from_sst(sst_df, grid_resolution=0.5)

# Or fetch all regions
all_data = fetcher.fetch_all_regions()
```

## Coastal Regions Included

- **Australia_East**: Queensland, NSW coastline
- **Australia_West**: Western Australia
- **USA_Florida**: Florida coast
- **USA_California**: California coast
- **South_Africa**: Cape region
- **Hawaii**: Hawaiian Islands
- **Brazil**: Brazilian coast
- **Mexico_Pacific**: Pacific Mexico coast

## Output Format

CSV files with columns:
- `region`: Region name
- `latitude`: Grid cell latitude (center)
- `longitude`: Grid cell longitude (center)
- `sst_mean`: Average sea surface temperature (°C)
- `sst_std`: Standard deviation of temperature
- `sst_min`: Minimum temperature
- `sst_max`: Maximum temperature
- `data_points`: Number of observations in this cell

## Data Sources

### Primary (Automated)
- NOAA OISST v2.1 (Optimum Interpolation Sea Surface Temperature)
- Access via ERDDAP servers

### Alternative (Manual Download)
If the automated fetch fails:
1. Visit https://www.ncei.noaa.gov/products/optimum-interpolation-sst
2. Download data for your regions
3. Place in `data/raw/` directory

## Notes

- Real data fetching may take several minutes
- NOAA servers have usage limits
- Sample data is generated if real data is unavailable
- Grid resolution is adjustable (default: 0.5 degrees)

## Integration with Shark Attack Model

```python
import pandas as pd

# Load SST data
sst_df = pd.read_csv('data/raw/sst_grid_sample.csv')

# Load shark attack data
attacks_df = pd.read_csv('data/raw/shark_attacks.csv')

# Merge by location (grid cells)
attacks_df['lat_grid'] = (attacks_df['latitude'] / 0.5).round() * 0.5
attacks_df['lon_grid'] = (attacks_df['longitude'] / 0.5).round() * 0.5

merged_df = attacks_df.merge(
    sst_df, 
    left_on=['lat_grid', 'lon_grid'],
    right_on=['latitude', 'longitude'],
    how='left'
)
```

## Dependencies

- requests
- pandas
- numpy

All included in `requirements.txt`.
