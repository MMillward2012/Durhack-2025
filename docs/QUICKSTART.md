# 🦈 Quick Start Guide - Shark Attack Prediction Heatmap
## Durham Hackathon 2025

---

## 🚀 Getting Started

Your project is set up! Here's what to do next:

### 1. Environment Check ✅
```bash
# Activate virtual environment
source .venv/bin/activate

# Check Python
python --version

# Test main script
python main.py
```

### 2. Install New Dependencies
```bash
# Install geospatial and mapping libraries
pip install -r requirements.txt
```

---

## 📊 Data Collection (Hour 0-3)

### Priority 1: Get GSAF Shark Attack Data
**Option A - Kaggle** (Easiest):
1. Go to Kaggle: https://www.kaggle.com/
2. Search for "shark attack" or "GSAF"
3. Download CSV (e.g., "Global Shark Attacks" dataset)
4. Save to `data/raw/shark_attacks.csv`

**Option B - GSAF Website**:
1. Visit: https://www.sharkattackfile.net/
2. Use their search/export feature
3. Save data manually

**Key columns needed**:
- Date
- Location (Country, State, City)
- Latitude/Longitude (if available)
- Activity (surfing, swimming, etc.)
- Fatal (Y/N)
- Species

### Priority 2: Ocean Temperature Data
**NOAA API** (Free):
1. Sign up: https://www.ncdc.noaa.gov/cdo-web/token
2. Get API token (arrives via email)
3. Add to `.env` file: `NOAA_API_KEY=your_key`

**Quick alternative**: Download sample SST data from NOAA website for your regions

### Priority 3: Tourism Estimates
**Quick approach**:
1. Google "beach visitor statistics [region name]"
2. Create simple CSV with estimates:
   ```csv
   location,peak_season,visitors_per_day
   Bondi Beach,Dec-Feb,40000
   Malibu,Jun-Aug,15000
   ```
3. Save to `data/raw/tourism_estimates.csv`

### Priority 4: Weather/Fish Data (Optional for MVP)
- Can add later if time permits
- Use proxies or simplified assumptions for MVP

---

## 🗺️ Project Structure

Create these directories if they don't exist:
```bash
mkdir -p data/raw data/processed models maps
```

### Recommended file organization:
```
data/
  raw/
    shark_attacks.csv          # GSAF data
    ocean_temperature.csv      # SST data  
    tourism_estimates.csv      # Visitor data
    regions_of_interest.geojson # Focus areas
  processed/
    cleaned_attacks.csv        # After preprocessing
    feature_matrix.csv         # ML-ready data
```

---

## 💻 Workflow Walkthrough

### Step 1: Data Exploration (Jupyter)
```bash
jupyter notebook
```

Create notebook: `notebooks/01_EDA.ipynb`

```python
import pandas as pd
import matplotlib.pyplot as plt
import folium

# Load shark attack data
df = pd.read_csv('../data/raw/shark_attacks.csv')

# Basic exploration
print(df.head())
print(df.columns)
print(df['Activity'].value_counts())

# Attacks by country
df['Country'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Countries by Shark Attacks')
plt.show()

# Simple map of attacks
m = folium.Map(location=[0, 0], zoom_start=2)
for idx, row in df.dropna(subset=['Latitude', 'Longitude']).iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color='red'
    ).add_to(m)
m.save('../maps/attack_locations.html')
```

### Step 2: Data Cleaning
```python
# Handle missing coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])

# Standardize dates
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Filter to modern era (better data quality)
df = df[df['Year'] >= 1990]

# Save cleaned data
df.to_csv('../data/processed/cleaned_attacks.csv', index=False)
```

### Step 3: Create Grid System
```python
import numpy as np

# Define coastal regions grid (0.5 degree cells)
lat_min, lat_max = -50, 50  # Focus on populated coasts
lon_min, lon_max = -180, 180

grid_size = 0.5
lats = np.arange(lat_min, lat_max, grid_size)
lons = np.arange(lon_min, lon_max, grid_size)

# Create grid cells
grid_cells = []
for lat in lats:
    for lon in lons:
        grid_cells.append({
            'cell_id': f"{lat}_{lon}",
            'lat_center': lat + grid_size/2,
            'lon_center': lon + grid_size/2,
            'attack_count': 0
        })

grid_df = pd.DataFrame(grid_cells)
```

### Step 4: Feature Engineering
```python
# Assign attacks to grid cells
def assign_to_grid(lat, lon, grid_size=0.5):
    grid_lat = int(lat / grid_size) * grid_size
    grid_lon = int(lon / grid_size) * grid_size
    return f"{grid_lat}_{grid_lon}"

df['cell_id'] = df.apply(
    lambda row: assign_to_grid(row['Latitude'], row['Longitude']), 
    axis=1
)

# Count attacks per cell
attack_counts = df['cell_id'].value_counts()

# Merge with grid
grid_df['attack_count'] = grid_df['cell_id'].map(attack_counts).fillna(0)

# Create target variable (high risk = 1+ attacks)
grid_df['high_risk'] = (grid_df['attack_count'] > 0).astype(int)

# Add features (simplified for MVP)
grid_df['latitude'] = grid_df['lat_center']
grid_df['longitude'] = grid_df['lon_center']
grid_df['distance_from_equator'] = abs(grid_df['latitude'])

# If you have temp data:
# grid_df = grid_df.merge(temp_df, on='cell_id', how='left')
```

### Step 5: Train Model
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare features
features = ['latitude', 'longitude', 'distance_from_equator']
X = grid_df[features]
y = grid_df['high_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Get predictions
grid_df['risk_score'] = model.predict_proba(X)[:, 1]
```

### Step 6: Create Heatmap
```python
import folium
from folium.plugins import HeatMap

# Create base map
m = folium.Map(location=[0, 0], zoom_start=2)

# Prepare heatmap data
heat_data = [[row['lat_center'], row['lon_center'], row['risk_score']] 
             for idx, row in grid_df.iterrows() if row['risk_score'] > 0.1]

# Add heatmap layer
HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

# Save map
m.save('../maps/risk_heatmap.html')
print("Heatmap saved! Open maps/risk_heatmap.html in browser")
```

---

## 🎨 Creating the Demo

### Option 1: Streamlit App (Recommended)
Create `app.py`:

```python
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static

st.title("🦈 Shark Attack Risk Heatmap")
st.write("Predicting shark attack risk zones using historical data")

# Load data
grid_df = pd.read_csv('data/processed/feature_matrix.csv')

# Sidebar filters
st.sidebar.header("Filters")
min_risk = st.sidebar.slider("Minimum Risk Score", 0.0, 1.0, 0.1)

# Filter data
filtered = grid_df[grid_df['risk_score'] >= min_risk]

# Create map
m = folium.Map(location=[0, 0], zoom_start=2)
for idx, row in filtered.iterrows():
    folium.CircleMarker(
        location=[row['lat_center'], row['lon_center']],
        radius=5,
        color='red' if row['risk_score'] > 0.5 else 'orange',
        fill=True,
        popup=f"Risk: {row['risk_score']:.2%}"
    ).add_to(m)

# Display map
folium_static(m)

# Show stats
st.subheader("Risk Statistics")
st.write(f"High risk zones: {len(filtered)}")
st.write(f"Average risk score: {filtered['risk_score'].mean():.2%}")
```

Run with:
```bash
streamlit run app.py
```

---

## ⏱️ Time Management Tips

### Hour 0-3: Data Collection
- Focus on GSAF data first (most important!)
- Get at least 100 attack records with locations
- Don't get stuck on perfect data - use estimates

### Hour 3-6: EDA
- Visualize attack locations
- Look for patterns (seasons, regions, activities)
- Document interesting findings

### Hour 6-10: Model Building
- Start simple: location-based risk only
- Add complexity if time permits
- Get something working early!

### Hour 10-14: Heatmap
- Basic folium map first
- Add interactivity later
- Make it visual and impressive

### Hour 14-18: Demo & Polish
- Streamlit is your friend (fast to build)
- Focus on presentation quality
- Practice your pitch!

---

## 🎯 MVP vs Stretch Goals

### Minimum Viable Product (4-6 hours)
- [ ] GSAF data loaded and cleaned
- [ ] Basic grid system created
- [ ] Simple model trained (just using location)
- [ ] Static heatmap generated
- [ ] Can explain approach clearly

### Good Demo (8-12 hours)
- [ ] Multiple data sources (GSAF + temp)
- [ ] Feature engineering (temporal, environmental)
- [ ] Random Forest or XGBoost model
- [ ] Interactive folium heatmap
- [ ] Streamlit demo app
- [ ] Presentation slides prepared

### Excellent/Stretch (12-24 hours)
- [ ] Real-time data integration
- [ ] Activity-specific risk maps
- [ ] Temporal predictions (seasonal changes)
- [ ] High model accuracy (>75%)
- [ ] Polished web app with multiple views
- [ ] Comprehensive documentation

---

## 🆘 Troubleshooting

**Problem**: Can't find GSAF data
- **Solution**: Use Kaggle "shark attacks" dataset, it's pre-formatted

**Problem**: Missing coordinates in data
- **Solution**: Use geocoding API or filter to records with lat/long

**Problem**: Too much data to process
- **Solution**: Focus on 3-4 regions (Australia, USA, South Africa)

**Problem**: Model accuracy is low
- **Solution**: That's OK! Focus on insights and visualization

**Problem**: Folium map not displaying
- **Solution**: Make sure to call `.save()` and open HTML file in browser

---

## 📚 Useful Resources

### Data Sources
- **GSAF**: https://www.sharkattackfile.net/
- **Kaggle Shark Attacks**: Search "shark attack dataset"
- **NOAA**: https://www.ncdc.noaa.gov/
- **OpenWeatherMap**: https://openweathermap.org/api

### Documentation
- **Folium**: https://python-visualization.github.io/folium/
- **GeoPandas**: https://geopandas.org/
- **Streamlit**: https://docs.streamlit.io/

### Example Notebooks
- Look for "geospatial machine learning" tutorials
- "Heatmap with folium" examples

---

## ✅ Quick Checklist

Before you start coding:
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] GSAF data downloaded
- [ ] Project structure created
- [ ] Jupyter notebook opened

Before demo time:
- [ ] Model trained and saved
- [ ] Heatmap generated
- [ ] Demo app tested
- [ ] Presentation prepared
- [ ] Code pushed to GitHub

---

**You're ready to go! Start with data collection and EDA. Good luck! 🦈🗺️**
