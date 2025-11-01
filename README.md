# 🦈 Shark Attack Prediction Heatmap
**Durham Hackathon 2025 - Predicting the Future Challenge**

## 🎯 What We're Building
A predictive **heatmap model** that forecasts shark attack risk zones by analyzing multiple environmental and human activity factors.

Can we predict where and when shark attacks are most likely to occur based on:
- **Historical Attack Data** (GSAF - Global Shark Attack File)
- **Ocean Temperature** (SST - Sea Surface Temperature)
- **Beach Tourism Levels** (visitor counts, seasonal trends)
- **Fish Migration Patterns** (prey availability)
- **Coastal Weather Conditions** (storms, visibility, water clarity)
- **Victim Activity Types** (surfing, swimming, diving, etc.)

## 🌊 The Problem
Shark attacks, while rare, have serious consequences. By predicting high-risk zones and times, we can:
- Help beach authorities issue timely warnings
- Inform tourists and water sports enthusiasts
- Guide lifeguard and patrol resource allocation
- Improve coastal safety planning

## 🗺️ The Solution
An interactive **risk heatmap** that visualizes predicted shark attack probability across coastal regions, updated based on real-time or seasonal data inputs.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Project
```bash
# Exploratory analysis
jupyter notebook notebooks/EDA.ipynb

# Run main pipeline
python main.py
```

## 📋 Project Structure
```
├── data/              
│   ├── raw/           # GSAF data, ocean temp, tourism data
│   └── processed/     # Cleaned & merged datasets
├── notebooks/         # Jupyter notebooks for EDA & visualization
├── src/               # Source code (data processing, models, mapping)
├── models/            # Trained ML models
├── maps/              # Generated heatmap outputs
├── PROJECT_PLAN.md    # Detailed 24-hour hackathon plan
└── requirements.txt   # Python dependencies
```

## 📖 Full Project Plan
**→ See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete 24-hour development timeline, data sources, and technical approach!**

## 🛠️ Tech Stack
- **Data Processing**: pandas, numpy
- **Geospatial**: geopandas, folium, plotly
- **ML**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn, plotly, folium
- **Weather/Ocean Data**: NOAA APIs, OpenWeather
- **Demo**: streamlit + interactive folium maps

## 📊 Data Sources
1. **GSAF (Global Shark Attack File)** - Historical attack records
2. **NOAA** - Sea surface temperature, weather data
3. **FishBase / Migration Data** - Fish movement patterns
4. **Tourism Statistics** - Beach visitor data by region/season
5. **Coastal Activity Data** - Water sports participation

## 👥 Team
Durham Hackathon 2025 participants working on the "Predict the Future" challenge!

## 📝 License
See LICENSE file for details.