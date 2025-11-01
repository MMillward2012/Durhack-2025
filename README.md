# 🦈 Shark Attack Prediction Heatmap
**Durham Hackathon 2025 - Predicting the Future Challenge**

---

## 🎯 Project Overview

A predictive **heatmap model** that forecasts shark attack risk zones by analyzing multiple environmental and human activity factors.

**Can we predict where and when shark attacks are most likely to occur?**

### Key Data Inputs
- 📊 Historical Attack Data (GSAF - Global Shark Attack File)
- 🌡️ Ocean Temperature (SST - Sea Surface Temperature)
- 🏖️ Beach Tourism Levels (visitor counts, seasonal trends)
- 🐟 Fish Migration Patterns (prey availability)
- ⛈️ Coastal Weather Conditions (storms, visibility, water clarity)
- 🏄 Victim Activity Types (surfing, swimming, diving, etc.)

### Impact
By predicting high-risk zones and times, we can:
- Help beach authorities issue timely warnings
- Inform tourists and water sports enthusiasts
- Guide lifeguard and patrol resource allocation
- Improve coastal safety planning

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Fetch Data
```bash
# Fetch ocean temperature data
python src/fetch_noaa_sst.py
```

### 3. Run Analysis
```bash
# Open Jupyter for exploration
jupyter notebook

# Or run main pipeline
python main.py
```

---

## � Project Structure
```
Durhack-2025/
├── data/
│   ├── raw/              # GSAF data, ocean temp, tourism data
│   └── processed/        # Cleaned & merged datasets
├── src/                  # Source code (data fetching, processing, models)
├── notebooks/            # Jupyter notebooks for EDA & visualization
├── models/               # Trained ML models
├── maps/                 # Generated heatmap outputs
├── docs/                 # Detailed documentation
│   ├── PROJECT_PLAN.md   # 24-hour hackathon timeline
│   ├── QUICKSTART.md     # Step-by-step getting started guide
│   └── README_SST_FETCHER.md  # NOAA data fetcher docs
└── requirements.txt      # Python dependencies
```

---

## 📖 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Step-by-step instructions, code snippets, and workflow
- **[Project Plan](docs/PROJECT_PLAN.md)** - Complete 24-hour timeline, data sources, and technical approach
- **[SST Data Fetcher](docs/README_SST_FETCHER.md)** - How to fetch ocean temperature data

---

## 🛠️ Tech Stack

**Data Processing**: pandas, numpy  
**Geospatial**: geopandas, folium, shapely, plotly  
**Machine Learning**: scikit-learn, xgboost  
**Visualization**: matplotlib, seaborn, plotly, folium  
**Data Sources**: NOAA APIs, GSAF, OpenWeather  
**Demo**: streamlit + interactive folium maps  

---

## 📊 Key Data Sources

1. **[GSAF](https://www.sharkattackfile.net/)** - Global Shark Attack File (historical records)
2. **[NOAA OISST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)** - Sea surface temperature data
3. **[Kaggle Datasets](https://www.kaggle.com/)** - Pre-formatted shark attack datasets
4. **Tourism Statistics** - Beach visitor data (regional tourism boards)
5. **Weather Data** - OpenWeatherMap, NOAA weather APIs

---

## 🎯 Getting Started Checklist

- [ ] Clone repository and activate virtual environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Download GSAF shark attack data from Kaggle
- [ ] Run `python src/fetch_noaa_sst.py` to get ocean temperature data
- [ ] Start with [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed workflow
- [ ] Follow [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the 24-hour timeline

---

## 👥 Team

Durham Hackathon 2025 participants  
**Challenge**: Predict the Future

---

## 📝 License

See [LICENSE](LICENSE) file for details.