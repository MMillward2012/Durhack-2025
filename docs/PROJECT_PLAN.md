# 🦈 Shark Attack Risk Predictor - Implementation Checklist

## Phase 1: Data Collection (Hours 0-6)

### Shark Attack Data
- [ ] Download Global Shark Attack File from ISAF/GSAF  
- [ ] Filter to unprovoked attacks only  
- [ ] Remove records with invalid/missing dates  
- [ ] Remove records with unclear locations  
- [ ] Geocode location names to lat/long  
- [ ] Standardize date formats (YYYY-MM-DD)  
- [ ] Save cleaned dataset as `shark_attacks_clean.csv`  
- [ ] Verify: 3,000-5,000 quality records  

### Ocean Temperature Data
- [ ] Access NOAA ERDDAP or Copernicus Marine Service  
- [ ] Define 50-100 prediction locations (lat/long)  
- [ ] Download historical monthly SST for each location  
- [ ] Calculate monthly averages (last 20 years)  
- [ ] Create lookup table: `{(location_id, month): temperature}`  
- [ ] Save as `temp_lookup.pkl` or `temp_lookup.json`  
- [ ] Verify: temperatures in reasonable range (10-35°C)  

### Migration Calendar
- [ ] Research seal pupping seasons by region  
- [ ] Research fish migration patterns by region  
- [ ] Create binary calendar: `{region: {month: 0 or 1}}`  
- [ ] Save as `migration_calendar.json`  
- [ ] Regions: California, Florida, Australia, South Africa, Hawaii  

### Tourism Patterns
- [ ] Define tourism intensity by region and month  
- [ ] Use hemisphere-based heuristics (summer = high)  
- [ ] Create lookup: `{region: {month: 0.0-1.0}}`  
- [ ] Save as `tourism_patterns.json`  

### Define Prediction Grid
- [ ] Create list of 50-100 coastal locations  
- [ ] Include: name, lat, long, region, country  
- [ ] Mix of famous beaches and shark hotspots  
- [ ] Ensure geographic diversity  
- [ ] Save as `prediction_locations.csv`  

---

## Phase 2: Data Processing (Hours 6-10)

### Feature Engineering for Historical Attacks
- [ ] For each attack, extract month from date  
- [ ] For each attack, look up ocean_temp from lookup table  
- [ ] For each attack, check migration_active from calendar  
- [ ] For each attack, look up tourism_level from patterns  
- [ ] Calculate is_weekend, is_summer  
- [ ] Calculate region_attack_frequency  
- [ ] Calculate days_since_last_attack in region  
- [ ] Label all with `attack = 1`  
- [ ] Save as `positive_samples.csv`  

### Generate Negative Samples
- [ ] Randomly sample 5-10x locations from prediction grid  
- [ ] Randomly sample dates from historical range  
- [ ] Verify no attack occurred at location+date  
- [ ] Extract same features as positive samples  
- [ ] Label all with `attack = 0`  
- [ ] Save as `negative_samples.csv`  

### Combine Training Data
- [ ] Merge positive and negative samples  
- [ ] Shuffle dataset  
- [ ] Check class distribution (10-20% positive)  
- [ ] Handle any missing values  
- [ ] Save as `training_data.csv`  
- [ ] Verify: 15,000-50,000 total rows  

### Train/Test Split
- [ ] Split by time: train on pre-2020, test on 2020-2024  
- [ ] Ensure no data leakage  
- [ ] Save as `train.csv` and `test.csv`  

---

## Phase 3: Model Training (Hours 10-14)

### Model Setup
- [ ] Import scikit-learn RandomForestClassifier  
- [ ] Define feature columns  
- [ ] Load training data  
- [ ] Separate X (features) and y (labels)  

### Train Model
- [ ] Initialize RandomForest(n_estimators=100-200)  
- [ ] Consider class_weight='balanced'  
- [ ] Fit model on training data  
- [ ] Save trained model as `shark_model.pkl`  

### Model Evaluation
- [ ] Load test data  
- [ ] Generate predictions on test set  
- [ ] Calculate accuracy, precision, recall  
- [ ] Generate confusion matrix  
- [ ] Calculate ROC-AUC score  
- [ ] Document performance metrics  

### Feature Importance Analysis
- [ ] Extract feature_importances_ from model  
- [ ] Rank features by importance  
- [ ] Create visualization of top 10 features  
- [ ] Save chart as `feature_importance.png`  
- [ ] Document surprising findings  

---

## Phase 4: Prediction System (Hours 14-18)

### Prediction Pipeline
- [ ] Create function: extract_features(location, date)  
- [ ] Function looks up temp from lookup table  
- [ ] Function checks migration calendar  
- [ ] Function gets tourism level  
- [ ] Function calculates temporal features  
- [ ] Create function: predict_risk(location, date)  
- [ ] Function returns risk score 0-100  

### Pre-compute Resources
- [ ] Load all lookup tables into memory  
- [ ] Test prediction speed (should be <100ms per location)  
- [ ] Optimize if necessary  

### Validation
- [ ] Test predictions on known high-risk scenarios  
- [ ] Test predictions on known low-risk scenarios  
- [ ] Verify seasonal patterns make sense  
- [ ] Document any issues  

---

## Phase 5: Visualization (Hours 18-22)

### Frontend Setup
- [ ] Choose framework (React recommended)  
- [ ] Set up project structure  
- [ ] Install dependencies (Plotly, mapping library)  

### World Map Component
- [ ] Create base world map  
- [ ] Plot 50-100 prediction locations  
- [ ] Color-code by risk level (green/yellow/red)  
- [ ] Add hover tooltips with location names  
- [ ] Scale marker size appropriately  

### Date Slider Component
- [ ] Create date range slider  
- [ ] Set date range (e.g., 2024-2026)  
- [ ] Display selected date clearly  
- [ ] Connect to prediction update function  

### Risk Calculation Integration
- [ ] On slider change, extract selected date  
- [ ] For each location, call predict_risk(location, date)  
- [ ] Update map colors based on new risk scores  
- [ ] Ensure smooth performance (<1 second update)  

### Location Detail Panel (Optional)
- [ ] Add click handler to map markers  
- [ ] Show popup with location details  
- [ ] Display current risk score  
- [ ] Show contributing factors:
  - Ocean temperature  
  - Migration season status  
  - Tourism level  
  - Historical attack frequency  

### Color Scale Legend
- [ ] Add legend showing risk levels  
- [ ] Green: 0-30 (Low)  
- [ ] Yellow: 30-70 (Moderate)  
- [ ] Red: 70-100 (High)  

### Polish
- [ ] Add loading states  
- [ ] Add error handling  
- [ ] Test on different screen sizes  
- [ ] Optimize performance  

---

## Phase 6: Analysis & Insights (Throughout)

### Find Surprising Patterns
- [ ] Analyze feature importance results  
- [ ] Look for interaction effects (temp × migration)  
- [ ] Compare different regions  
- [ ] Examine temporal trends  
- [ ] Check species-specific patterns (if data allows)  

### Document Key Findings
- [ ] Finding #1: (e.g., "Migration season increases risk 400%")  
- [ ] Finding #2: (e.g., "Temperature threshold at 22°C")  
- [ ] Finding #3: (e.g., "Morning attacks 3x more common")  
- [ ] Create visualizations for each finding  

---

## Phase 7: Presentation (Hours 22-24)

### Slide Deck
- [ ] Title slide with project name  
- [ ] Problem statement (30 seconds)  
- [ ] Data overview (1 minute)  
- [ ] Live demo preparation (2 minutes)  
- [ ] Key insights (1 minute)  
- [ ] Technical details (backup slides)  
- [ ] Future work (backup slides)  

### Demo Preparation
- [ ] Test live demo flow  
- [ ] Prepare specific dates to showcase  
- [ ] Identify interesting locations to highlight  
- [ ] Practice transitions  
- [ ] Have backup screenshots if demo fails  

### Story Arc
- [ ] Opening hook about shark attacks  
- [ ] Show the interactive map  
- [ ] Drag slider to show seasonal changes  
- [ ] Click location to show details  
- [ ] Present surprising findings  
- [ ] Discuss practical applications  

### Practice
- [ ] Run through presentation 2-3 times  
- [ ] Time each section  
- [ ] Prepare for Q&A  
- [ ] Test demo on presentation laptop  

---

## Key Deliverables Checklist

### Data Files
- [ ] `shark_attacks_clean.csv`  
- [ ] `prediction_locations.csv`  
- [ ] `temp_lookup.json` or `.pkl`  
- [ ] `migration_calendar.json`  
- [ ] `tourism_patterns.json`  
- [ ] `training_data.csv`  
- [ ] `train.csv` and `test.csv`  

### Model Files
- [ ] `shark_model.pkl`  
- [ ] `feature_importance.png`  
- [ ] `model_metrics.txt`  

### Application
- [ ] Interactive web application  
- [ ] Working date slider  
- [ ] Real-time risk predictions  
- [ ] Visual heatmap  

### Presentation
- [ ] Slide deck (PDF or PowerPoint)  
- [ ] Demo video (backup)  
- [ ] GitHub repository with README  

---

## Emergency Contingencies

### If Running Out of Time
**Priority 1 (Must Have):**
- [ ] Working model with predictions  
- [ ] Basic map visualization  
- [ ] Date slider functionality  

**Priority 2 (Should Have):**
- [ ] Feature importance analysis  
- [ ] 1-2 key insights identified  
- [ ] Polished presentation  

**Priority 3 (Nice to Have):**
- [ ] Click interactions  
- [ ] Animation mode  
- [ ] Detailed tooltips  

### If Data Issues
- [ ] Reduce prediction locations to 25-50  
- [ ] Use simplified temperature estimates  
- [ ] Focus on 1-2 regions with best data  
- [ ] Simplify migration calendar  

### If Model Performance Poor
- [ ] Frame as "risk indicator" not "predictor"  
- [ ] Focus on visualization and interactivity  
- [ ] Emphasize data integration novelty  
- [ ] Show feature importance regardless of accuracy  