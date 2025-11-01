# 🦈 Shark Attack Prediction Heatmap - Project Plan
## Durham Hackathon 2025

---

## 🎯 Project Goal
**Predict shark attack risk zones using historical data and environmental factors to create an interactive heatmap.**

Build a predictive model that identifies high-risk areas and times for shark attacks based on:
- Historical shark attack patterns (GSAF database)
- Ocean conditions (temperature, currents)
- Human activity levels (beach tourism, water sports)
- Ecological factors (fish migration, prey availability)
- Weather conditions (storms, visibility)

---

## 💡 Why This is Novel & Impactful

### Novelty
- **Multi-factor prediction**: Most shark analysis is reactive; we're predicting future risk
- **Geospatial ML**: Combining location data with temporal and environmental features
- **Real-world utility**: Actually deployable for beach safety
- **Data fusion**: Merging ecological, weather, tourism, and historical data

### Impact
- **Public Safety**: Reduce shark attack incidents through early warnings
- **Tourism Management**: Help beach authorities allocate resources
- **Conservation**: Better understanding of shark-human interaction patterns
- **Evidence-based policy**: Data-driven coastal management decisions

---

## 📊 Data Sources & Features

### 1. **Historical Shark Attack Data (GSAF)**
**Source**: Global Shark Attack File - https://www.sharkattackfile.net/
- Attack locations (lat/long)
- Date and time
- Victim activity (surfing, swimming, diving, fishing)
- Shark species (if identified)
- Injury severity
- Water conditions at time of attack

### 2. **Ocean Temperature (SST - Sea Surface Temperature)**
**Sources**: 
- NOAA (https://www.ncei.noaa.gov/)
- Copernicus Marine Service
- OpenWeatherMap Ocean API

**Features**:
- Current SST by region
- Temperature anomalies
- Seasonal temperature patterns
- Thermocline depth

### 3. **Beach Tourism Data**
**Sources**:
- Local tourism boards
- Google Popular Times API (for beaches)
- Social media check-in data
- Beach visitor statistics

**Features**:
- Daily/weekly visitor counts
- Peak season vs off-season
- Holiday periods
- Local events

### 4. **Fish Migration & Prey Availability**
**Sources**:
- FishBase (https://www.fishbase.org/)
- Ocean Biogeographic Information System (OBIS)
- Regional fisheries data

**Features**:
- Seasonal fish migrations
- Baitfish abundance
- Spawning seasons
- Marine protected areas

### 5. **Coastal Weather Conditions**
**Sources**:
- NOAA Weather API
- OpenWeatherMap
- Local meteorological services

**Features**:
- Wave height and conditions
- Water visibility
- Storm activity
- Wind speed/direction
- Tidal patterns

### 6. **Victim Activity Classification**
**From GSAF data**:
- Surfing/bodyboarding
- Swimming/wading
- Diving/snorkeling
- Fishing
- Kayaking/paddleboarding
- Standing in water

---

## 🔬 Technical Approach

### Phase 1: Data Collection & Preprocessing
1. **Scrape/download GSAF data** (primary source)
2. **Fetch ocean temperature data** (NOAA API)
3. **Collect tourism statistics** (manual + APIs)
4. **Get fish migration data** (databases)
5. **Weather data collection** (APIs)
6. **Geospatial data cleanup** (standardize coordinates)

### Phase 2: Feature Engineering
Create risk factors:
- **Temporal features**: month, season, time of day, day of week
- **Ocean features**: SST, temp anomaly, thermocline
- **Activity risk scores**: historical attack rates by activity type
- **Tourism density**: visitor count normalization
- **Ecological features**: prey availability score
- **Weather risk**: visibility, wave height, storm proximity
- **Historical risk**: past attacks in region (spatial clustering)

### Phase 3: Model Development
**Approach**: Geospatial risk prediction

**Model Options**:
1. **Grid-based Classification**:
   - Divide coastal regions into grid cells
   - Train classifier for "high risk" vs "low risk" per cell
   - Models: Random Forest, XGBoost, Logistic Regression

2. **Density-based Prediction**:
   - Kernel Density Estimation (KDE) on historical attacks
   - Weighted by environmental conditions
   - Output: probability heatmap

3. **Time-series Forecasting** (stretch):
   - Predict risk levels for future dates
   - LSTM or Prophet for temporal patterns

**Target Variable**:
- Binary: High risk (1) vs Low risk (0) per grid cell
- Or continuous: Attack probability score (0-1)

### Phase 4: Visualization
**Interactive Heatmap**:
- **Base map**: Folium or Plotly
- **Heat layer**: Color-coded risk zones
- **Interactive filters**: Date range, activity type, region
- **Info popups**: Risk factors for each zone
- **Time slider**: Show risk changes over seasons

**Additional Visualizations**:
- Attack frequency by month/season
- Risk correlation with temperature
- Activity-specific risk maps
- Historical vs predicted overlays

---

## 📅 24-Hour Development Timeline

### **Hour 0-3: Data Collection & Setup**
**Goal**: Get all data sources loaded

- [ ] Download GSAF dataset (CSV from website or Kaggle)
- [ ] Sign up for NOAA API access
- [ ] Sign up for OpenWeatherMap API (free tier)
- [ ] Create data directory structure
- [ ] Initial data exploration in Jupyter
- [ ] Identify key coastal regions to focus on (e.g., Australia, California, Florida, South Africa)

**Deliverable**: `data/raw/` folder with GSAF, SST samples, tourism estimates

---

### **Hour 3-6: Data Cleaning & EDA**
**Goal**: Understand patterns, clean data

- [ ] Clean GSAF data (handle missing values, standardize locations)
- [ ] Geocode any text-based locations to lat/long
- [ ] Merge datasets by location and date
- [ ] Exploratory visualizations:
  - Attack frequency by region
  - Seasonal patterns
  - Activity type distribution
  - Temperature correlation
- [ ] Identify data quality issues

**Deliverable**: `notebooks/EDA.ipynb` with insights

---

### **Hour 6-10: Feature Engineering & Grid Creation**
**Goal**: Build feature matrix for ML

- [ ] Create coastal grid system (e.g., 0.5° x 0.5° cells)
- [ ] Assign attacks to grid cells
- [ ] Calculate features per grid cell:
  - Historical attack count
  - Average SST
  - Tourism density estimate
  - Prey availability score
  - Weather risk factors
- [ ] Create labeled dataset (high risk = cells with attacks)
- [ ] Handle class imbalance (many cells have 0 attacks)

**Deliverable**: `data/processed/feature_matrix.csv`

---

### **Hour 10-14: Model Training & Validation**
**Goal**: Build predictive model

- [ ] Split data: 70% train, 15% validation, 15% test
- [ ] Train baseline model (Logistic Regression)
- [ ] Train advanced models (Random Forest, XGBoost)
- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Evaluate metrics:
  - Precision/Recall (important for safety!)
  - ROC-AUC
  - Confusion matrix
- [ ] Feature importance analysis
- [ ] Model interpretation

**Deliverable**: `models/shark_risk_model.pkl`, performance report

---

### **Hour 14-18: Heatmap Visualization**
**Goal**: Create interactive map

- [ ] Install/setup Folium or Plotly
- [ ] Generate base coastal map
- [ ] Overlay risk predictions as heatmap
- [ ] Add interactive elements:
  - Click on zone for details
  - Filter by season/month
  - Toggle different risk factors
- [ ] Create static visualizations for presentation
- [ ] Export high-quality images

**Deliverable**: Interactive HTML map, static charts

---

### **Hour 18-22: Demo Application & Polish**
**Goal**: Finalize presentation

- [ ] Build Streamlit demo app:
  - Upload location
  - Select date/season
  - Choose activity type
  - See risk prediction + heatmap
- [ ] Add example case studies
- [ ] Create presentation slides
- [ ] Document methodology
- [ ] Clean up code
- [ ] Write clear README

**Deliverable**: `app.py`, presentation slides

---

### **Hour 22-24: Testing, Practice & Buffer**
**Goal**: Ready for judging

- [ ] Test demo end-to-end
- [ ] Prepare 3-5 minute pitch
- [ ] Practice presentation
- [ ] Prepare for Q&A
- [ ] Final GitHub push
- [ ] Backup demo (screenshots/video)

**Deliverable**: Polished demo, practiced pitch

---

## 🛠️ Tech Stack Details

### Core Libraries
```python
# Data processing
pandas
numpy
geopandas

# Machine Learning
scikit-learn
xgboost

# Geospatial & Mapping
folium
plotly
shapely
pyproj

# Data visualization
matplotlib
seaborn

# APIs
requests
python-dotenv

# Web demo
streamlit
```

### API Keys Needed
- NOAA Climate Data API (free)
- OpenWeatherMap API (free tier)
- Optional: Google Maps API (for geocoding)

---

## 📦 Project Structure

```
Durhack-2025/
├── data/
│   ├── raw/
│   │   ├── gsaf_attacks.csv          # Historical shark attacks
│   │   ├── ocean_temperature.csv     # SST data
│   │   ├── tourism_estimates.csv     # Beach visitor data
│   │   └── fish_migration.csv        # Prey availability
│   └── processed/
│       ├── feature_matrix.csv        # ML-ready features
│       └── grid_cells.geojson        # Coastal grid system
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── data_collection.py            # API calls, scraping
│   ├── preprocessing.py              # Data cleaning
│   ├── feature_engineering.py        # Create features
│   ├── grid_system.py                # Geospatial grid
│   ├── model.py                      # ML model
│   └── visualization.py              # Heatmap generation
├── models/
│   └── shark_risk_model.pkl          # Trained model
├── maps/
│   └── risk_heatmap.html             # Interactive map
├── app.py                            # Streamlit demo
├── .env.example                      # API key template
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md                   # This file
```

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)
- [ ] Historical attack data collected and cleaned
- [ ] Basic risk model trained (>60% accuracy)
- [ ] Static heatmap visualization created
- [ ] Can predict risk for at least 3 major coastal regions
- [ ] Clear presentation explaining approach

### Stretch Goals
- [ ] Interactive web demo with real-time predictions
- [ ] Multiple activity-specific risk maps
- [ ] Temporal predictions (seasonal risk changes)
- [ ] >75% model accuracy with good precision/recall
- [ ] Published dataset and model on GitHub

---

## 📝 Model Evaluation Strategy

### Metrics to Track
1. **Precision**: Of predicted high-risk zones, how many actually had attacks?
2. **Recall**: Of actual attacks, how many occurred in predicted high-risk zones?
3. **F1-Score**: Balance of precision and recall
4. **Spatial Accuracy**: Are predictions geographically sensible?

### Validation Approach
- **Temporal split**: Train on older data, test on recent years
- **Geographic split**: Train on some regions, test on others
- **Cross-validation**: K-fold with spatial awareness

---

## 🎤 Presentation Tips

### Demo Flow (3-5 minutes)
1. **Hook** (30s): "Shark attacks are rare but devastating. What if we could predict where they'll happen?"
2. **Problem** (30s): Current approach is reactive; we want predictive safety
3. **Data** (45s): Show the multi-source data (GSAF, ocean temp, tourism, etc.)
4. **Model** (45s): Explain how we predict risk zones
5. **Demo** (90s): Show interactive heatmap, pick a location, explain prediction
6. **Impact** (30s): Beach safety, resource allocation, evidence-based warnings

### Key Talking Points
- **Novelty**: Multi-factor geospatial prediction (not just historical clustering)
- **Accuracy**: Model performance metrics
- **Interpretability**: Show feature importance (temperature matters!)
- **Usability**: Beach authorities can use this for real safety decisions
- **Cool factor**: Interactive map is visually impressive

### Handling Questions
- **"How accurate is it?"** → Focus on precision/recall for high-risk zones
- **"What data did you use?"** → List sources, mention limitations
- **"Is this deployable?"** → Yes, with more data and validation
- **"What about false alarms?"** → Discuss precision-recall tradeoff

---

## 🚨 Potential Challenges & Solutions

### Challenge: Limited historical attack data
**Solution**: 
- Use all available GSAF data (1900s to present)
- Augment with near-miss data if available
- Use data augmentation techniques

### Challenge: Tourism data hard to find
**Solution**:
- Use proxies: population density, beach ratings, social media
- Manual estimates for popular beaches
- Focus on relative differences (high vs low tourism)

### Challenge: Model might be overfitted to historical data
**Solution**:
- Use regularization
- Test on recent years
- Focus on generalizable features (temp, season)

### Challenge: Real-time data integration is complex
**Solution**:
- Start with static predictions by season
- Demo with "what-if" scenarios
- Mention real-time as future work

---

## 🔍 Data Collection Specifics

### GSAF Data
**Website**: https://www.sharkattackfile.net/
**Alternative**: Kaggle - search "shark attack dataset"

**Key fields**:
- Date
- Country/State/Location
- Activity
- Species
- Fatal/Non-fatal

### NOAA Ocean Temperature
**API**: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
**Free tier**: 1000 requests/day
**Data**: Sea Surface Temperature (SST) by coordinates

### Tourism Data
**Sources**:
- Tourism agency reports (PDF → manual entry)
- TripAdvisor beach ratings/review counts
- Google Popular Times (if accessible)
- Wikipedia beach traffic estimates

---

## ✅ Pre-Submission Checklist

- [ ] Code runs without errors
- [ ] Heatmap displays correctly
- [ ] Model performance documented
- [ ] README explains project clearly
- [ ] GitHub repo is organized and clean
- [ ] Demo is rehearsed and timed
- [ ] All team members can explain the approach
- [ ] API keys are in `.env` (not committed!)
- [ ] Data sources are cited

---

## 🏆 Why This Will Impress Judges

1. **Real-world impact**: Actual safety application
2. **Technical depth**: Geospatial ML + multiple data sources
3. **Visual appeal**: Interactive heatmap is eye-catching
4. **Novelty**: Predictive, not just descriptive
5. **Presentation**: Clear problem → solution → demo flow
6. **Completeness**: Data + model + visualization + demo

---

Good luck! 🦈🗺️ You've got a fantastic project idea!

**Remember**: Focus on getting a working MVP first, then add polish. A simple working demo beats a complex broken one every time!
