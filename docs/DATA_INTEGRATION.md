# Data Integration Guide

How to merge shark attack data with ocean temperature and other environmental factors.

## 📊 Available Data Files

### 1. SST (Sea Surface Temperature) Data
**Location**: `data/raw/sst_grid_sample_with_dates.csv`

**Columns**:
- `region`: Coastal region name
- `latitude`, `longitude`: Grid cell center
- `date`: Date of measurement (YYYY-MM-DD)
- `year`, `month`, `day`: Date components
- `sst_mean`: Average temperature (°C)
- `sst_std`: Temperature standard deviation
- `sst_min`, `sst_max`: Temperature range
- `data_points`: Number of observations

### 2. Shark Attack Data (GSAF)
**Location**: `data/raw/shark_attacks.csv` (to be downloaded)

**Expected columns**:
- `Date`: Attack date
- `Country`, `State`, `Location`: Location text
- `Latitude`, `Longitude`: Coordinates
- `Activity`: What victim was doing
- `Species`: Shark species (if known)
- `Fatal`: Y/N

---

## 🔄 Data Merging Workflow

### Step 1: Prepare Shark Attack Data

```python
import pandas as pd
import numpy as np

# Load shark attack data
attacks_df = pd.read_csv('data/raw/shark_attacks.csv')

# Clean and parse dates
attacks_df['date'] = pd.to_datetime(attacks_df['Date'], errors='coerce')
attacks_df['year'] = attacks_df['date'].dt.year
attacks_df['month'] = attacks_df['date'].dt.month
attacks_df['day'] = attacks_df['date'].dt.day

# Remove rows with missing coordinates or dates
attacks_df = attacks_df.dropna(subset=['Latitude', 'Longitude', 'date'])

# Create grid cell IDs to match SST data (0.5 degree resolution)
grid_resolution = 0.5
attacks_df['lat_grid'] = (attacks_df['Latitude'] / grid_resolution).round() * grid_resolution
attacks_df['lon_grid'] = (attacks_df['Longitude'] / grid_resolution).round() * grid_resolution

print(f"Loaded {len(attacks_df)} shark attacks with valid coordinates")
```

### Step 2: Load SST Data

```python
# Load SST data with dates
sst_df = pd.read_csv('data/raw/sst_grid_sample_with_dates.csv')

# Convert date column to datetime
sst_df['date'] = pd.to_datetime(sst_df['date'])

print(f"Loaded {len(sst_df)} SST data points")
```

### Step 3: Merge by Location AND Date

```python
# Exact date match
merged_df = attacks_df.merge(
    sst_df,
    left_on=['lat_grid', 'lon_grid', 'date'],
    right_on=['latitude', 'longitude', 'date'],
    how='left',
    suffixes=('_attack', '_sst')
)

# Check how many attacks got SST data
matched = merged_df['sst_mean'].notna().sum()
print(f"Matched {matched}/{len(merged_df)} attacks with SST data")
```

### Step 4: Handle Missing Dates (Optional)

If exact date matching doesn't work well, match by location and month:

```python
# Average SST by location and month
sst_monthly = sst_df.groupby(['latitude', 'longitude', 'month']).agg({
    'sst_mean': 'mean',
    'sst_std': 'mean',
    'sst_min': 'min',
    'sst_max': 'max'
}).reset_index()

# Merge by location and month
merged_df = attacks_df.merge(
    sst_monthly,
    left_on=['lat_grid', 'lon_grid', 'month'],
    right_on=['latitude', 'longitude', 'month'],
    how='left'
)
```

---

## 🎯 Complete Integration Example

```python
import pandas as pd
import numpy as np

# === 1. Load Shark Attacks ===
attacks = pd.read_csv('data/raw/shark_attacks.csv')
attacks['date'] = pd.to_datetime(attacks['Date'], errors='coerce')
attacks = attacks.dropna(subset=['Latitude', 'Longitude', 'date'])

# Create grid cells
attacks['lat_grid'] = (attacks['Latitude'] / 0.5).round() * 0.5
attacks['lon_grid'] = (attacks['Longitude'] / 0.5).round() * 0.5
attacks['year'] = attacks['date'].dt.year
attacks['month'] = attacks['date'].dt.month

# === 2. Load SST Data ===
sst = pd.read_csv('data/raw/sst_grid_sample_with_dates.csv')
sst['date'] = pd.to_datetime(sst['date'])

# === 3. Merge ===
# Try exact date match first
merged = attacks.merge(
    sst[['latitude', 'longitude', 'date', 'sst_mean', 'sst_std']],
    left_on=['lat_grid', 'lon_grid', 'date'],
    right_on=['latitude', 'longitude', 'date'],
    how='left'
)

# For missing SST, use monthly averages
sst_monthly = sst.groupby(['latitude', 'longitude', 'month'])['sst_mean'].mean().reset_index()
sst_monthly.rename(columns={'sst_mean': 'sst_monthly_avg'}, inplace=True)

merged = merged.merge(
    sst_monthly,
    left_on=['lat_grid', 'lon_grid', 'month'],
    right_on=['latitude', 'longitude', 'month'],
    how='left',
    suffixes=('', '_monthly')
)

# Fill missing exact SST with monthly average
merged['sst_final'] = merged['sst_mean'].fillna(merged['sst_monthly_avg'])

# === 4. Add Additional Features ===
# Tourism proxy (binary: peak season = summer months)
merged['is_peak_season'] = merged['month'].isin([6, 7, 8, 12, 1, 2]).astype(int)

# Activity risk categories
high_risk_activities = ['Surfing', 'Swimming', 'Diving']
merged['activity_risk'] = merged['Activity'].isin(high_risk_activities).astype(int)

# === 5. Save Merged Dataset ===
merged.to_csv('data/processed/attacks_with_features.csv', index=False)
print(f"✓ Saved {len(merged)} attacks with environmental features")
```

---

## 📈 Feature Engineering Tips

### Temporal Features
```python
# Season
merged['season'] = pd.cut(
    merged['month'], 
    bins=[0, 3, 6, 9, 12],
    labels=['Winter', 'Spring', 'Summer', 'Fall']
)

# Day of week (if you have daily data)
merged['day_of_week'] = merged['date'].dt.dayofweek
merged['is_weekend'] = merged['day_of_week'].isin([5, 6]).astype(int)
```

### Temperature Features
```python
# Temperature anomaly (deviation from regional average)
regional_avg = merged.groupby(['lat_grid', 'lon_grid'])['sst_final'].transform('mean')
merged['sst_anomaly'] = merged['sst_final'] - regional_avg

# Temperature categories
merged['temp_category'] = pd.cut(
    merged['sst_final'],
    bins=[0, 15, 20, 25, 35],
    labels=['cold', 'cool', 'warm', 'hot']
)
```

### Spatial Features
```python
# Distance from equator (proxy for climate)
merged['dist_from_equator'] = merged['Latitude'].abs()

# Coastal region clusters
from sklearn.cluster import KMeans
coords = merged[['lat_grid', 'lon_grid']].values
kmeans = KMeans(n_clusters=10, random_state=42)
merged['region_cluster'] = kmeans.fit_predict(coords)
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Missing SST Data
**Problem**: Not all attack locations have SST data  
**Solutions**:
1. Use nearest neighbor matching (find closest grid cell)
2. Use monthly/seasonal averages instead of daily
3. Interpolate missing values

```python
# Nearest neighbor fill
from scipy.spatial import cKDTree

# Build tree of SST coordinates
sst_coords = sst[['latitude', 'longitude']].values
tree = cKDTree(sst_coords)

# Find nearest SST cell for each attack
attack_coords = merged[['lat_grid', 'lon_grid']].values
distances, indices = tree.query(attack_coords, k=1)

# Assign nearest SST value
merged['sst_nearest'] = sst.iloc[indices]['sst_mean'].values
```

### Issue 2: Date Mismatches
**Problem**: SST dates don't exactly match attack dates  
**Solution**: Match by month or use time windows

```python
# Match within ±7 days
from datetime import timedelta

def find_nearby_sst(row, sst_df):
    location_match = (sst_df['latitude'] == row['lat_grid']) & \
                     (sst_df['longitude'] == row['lon_grid'])
    date_window = (sst_df['date'] >= row['date'] - timedelta(days=7)) & \
                  (sst_df['date'] <= row['date'] + timedelta(days=7))
    matches = sst_df[location_match & date_window]
    return matches['sst_mean'].mean() if len(matches) > 0 else np.nan

merged['sst_weekly'] = merged.apply(lambda row: find_nearby_sst(row, sst), axis=1)
```

---

## 📊 Final Dataset Structure

After merging, your dataset should have:

**Spatial**:
- `Latitude`, `Longitude`: Original coordinates
- `lat_grid`, `lon_grid`: Grid cell IDs
- `region`: Coastal region name

**Temporal**:
- `date`: Attack date
- `year`, `month`, `day`: Date components
- `season`: Season category

**Environmental**:
- `sst_mean`: Sea surface temperature
- `sst_anomaly`: Temperature deviation
- `is_peak_season`: Tourism indicator

**Attack Details**:
- `Activity`: Victim activity
- `Species`: Shark species
- `Fatal`: Attack severity

**Target Variable** (for ML):
- `occurred`: 1 (for all attacks in dataset)
- You'll need to generate negative samples (no attacks) for training!

---

## 🎯 Next Steps

1. **Generate negative samples**: Create random location-date combinations where NO attacks occurred
2. **Feature engineering**: Add more environmental factors (tourism, fish migration, weather)
3. **Train model**: Use merged data to predict attack risk
4. **Create heatmap**: Visualize predictions on a map

See `docs/QUICKSTART.md` for model training examples!
