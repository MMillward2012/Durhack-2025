# 🦈 Shark Attack Prediction Heatmap
**Durham Hackathon 2025 · Predict the Future Challenge**

Interactive modelling, prediction, and visualization tooling that forecasts global shark attack risk by combining environmental signals, human activity, and historical incident data. The project delivers both a reproducible data science pipeline and a production-ready Next.js web experience with 3D and 2D risk exploration.

---

## 🌍 What the Project Delivers

- **Data engineering pipeline** that aggregates NOAA SST data, population density, shark density grids, and historical incidents into training-ready features.
- **Machine learning models** (XGBoost) that estimate attack likelihood per location and month, including climate-adjusted post-processing.
- **Automated dataset generation** (`generate_webapp_data_simple.py`) that exports monthly heatmap JSON files consumed by the webapp.
- **Immersive web application** featuring a Cesium-powered 3D globe and Leaflet 2D satellite map with synchronized heatmap overlays, timeline scrubbing, and location-level risk callouts.
- **Actionable insights** for coastal authorities, lifeguards, and ocean users through dynamic high-risk zone detection and contextual climate metrics.

---

## 🏗️ Architecture at a Glance

| Layer | Highlights |
| --- | --- |
| **Data + Features** | Scripts in `src/` and the root helper utilities fetch SST climatology, population, and shark density, then engineer globe-wide samples per month. |
| **Modelling** | `models/` stores trained XGBoost artifacts used by `generate_webapp_data_simple.py` and `src/prediction.py` for inference and reporting. |
| **Visualization** | `webapp/` (Next.js + Cesium + Leaflet + Tailwind) renders the ML outputs as an interactive heatmap with timeline playback and risk drill-down. |

---

## 📂 Repository Structure

```
Durhack-2025/
├── data/
│   ├── raw/                 # Source CSVs (SST, shark incidents, etc.)
│   └── processed/           # Generated shark density grids & merged datasets
├── models/                  # Trained XGBoost model artifacts & evaluation plots
├── notebooks/               # Exploratory analysis and prototype modelling
├── src/                     # Python feature engineering + training utilities
│   ├── train_xgboost_model.py
│   ├── prediction.py        # Batch prediction / report generator
│   └── get_sst_data.py, get_pop_data.py, ...
├── webapp/                  # Next.js app (3D globe + 2D map UI)
│   ├── public/data/         # Heatmap JSON tiles (generated monthly)
│   └── src/app/...          # React components: Globe, LeafletMap, sidebar
├── generate_webapp_data_simple.py  # Main pipeline to produce webapp datasets
├── download_sst_data.py             # Helper fetcher for SST climatology
├── main.py                           # Entry banner / CLI helper
├── docs/                     # Hackathon plan & supplemental documentation
│   ├── PROJECT_PLAN.md
│   ├── QUICKSTART.md
│   └── README_SST_FETCHER.md
└── requirements.txt          # Python dependencies
```

---

## � Quick Start

### 1. Data & Modelling (Python)

```bash
# create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (optional) pull latest SST climatology
python download_sst_data.py

# generate monthly risk datasets for the webapp
python generate_webapp_data_simple.py

# retrain or evaluate the ML model
python src/train_xgboost_model.py
python src/prediction.py
```

Generated heatmap JSON files are written to `webapp/public/data/heatmap_YYYY_MM.json` and are immediately picked up by the frontend.

### 2. Interactive Web Experience (Next.js)

```bash
cd webapp
npm install
npm run dev
```

Navigate to `http://localhost:3000` to explore:

| Feature | Description |
| --- | --- |
| 3D Globe | Cesium-based Earth with dynamic heatmap overlay, climate stats, and click-to-inspect risk. |
| 2D Map | Leaflet satellite basemap with synchronized heatmap, respecting probability transparency thresholds. |
| Timeline Player | Scrub month-by-month to see risk evolution (view choice persists across timeline changes). |
| Sidebar Insights | Peak/average risk, climate adjustments, top-level metrics, and explanatory copy. |
| Detail Card | Clicking any location reveals latitude, longitude, and relative risk level. |

---

## 📚 Documentation & Supporting Material

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** – step-by-step workshop notes and CLI walkthroughs.
- **[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)** – the 24-hour hackathon timeline, milestones, and task allocation.
- **[docs/README_SST_FETCHER.md](docs/README_SST_FETCHER.md)** – NOAA SST data acquisition guide.

---

## 📊 Data Sources

1. **[Global Shark Attack File (GSAF)](https://www.sharkattackfile.net/)** – historical incident records.
2. **[NOAA Optimum Interpolation SST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)** – sea surface temperature baselines.
3. **Population & Tourism Datasets** – regional visitor counts and population proxies (`data/population/`).
4. **Derived Shark Density Grid** – processed sightings density stored in `data/processed/shark_density_grid.csv`.

---

## 🧰 Tech Stack

- **Python & Data Science**: pandas, numpy, scipy, scikit-learn, xgboost, joblib, geopy.
- **Geospatial Processing**: shapely, geopandas (notebooks), custom grid smoothing, Gaussian filters.
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, CesiumJS, React-Leaflet, ESRI World Imagery tiles.
- **Tooling**: Jupyter Notebooks, npm, Node.js, GitHub Actions (optional).

---

## 🤝 Contributing & Next Steps

Pull requests and experiments are welcome—consider extending the data ingestion pipeline, adding model explainability, or deploying the webapp. Review the existing documentation in `docs/` before proposing major changes.

---

## � License

This project is released under the terms of the [MIT License](LICENSE).