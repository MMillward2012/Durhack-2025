# Tweet Polarization Predictor - Project Plan
## Durham Hackathon 2025

---

## 🎯 Project Goal
**Predict whether a topic will become polarized or die off based on the first N tweets (20, 50, or 100).**

This is a novel approach to understanding social media dynamics by analyzing early signals in tweet conversations to forecast whether a topic will:
- **Polarize**: Generate strong opposing viewpoints and heated debate
- **Die Off**: Fade away without gaining traction or engagement

---

## 💡 Why This is Novel & Creative
- **Early Detection**: Most sentiment analysis looks at trends retrospectively. We're predicting the *future* trajectory.
- **Small Sample Size**: Using just the first 20-100 tweets makes this a challenging ML problem.
- **Real-World Impact**: Could help brands, journalists, or platforms identify controversial topics early.
- **Multi-dimensional Analysis**: Combines NLP, sentiment analysis, network effects, and temporal patterns.

---

## 📊 Project Scope

### Core Features (Must-Have)
1. **Data Collection**: Gather sample tweets on various topics (use Twitter API or pre-collected datasets)
2. **Feature Engineering**: Extract meaningful signals from early tweets
3. **Model Training**: Build a classifier to predict polarization vs. die-off
4. **Visualization**: Create an engaging demo showing predictions

### Stretch Goals (Nice-to-Have)
- Real-time Twitter API integration
- Interactive web dashboard
- Confidence scores and explanation of predictions
- Historical validation (show past predictions that came true)

---

## 🔬 Technical Approach

### Data Features to Extract
From the first N tweets, we'll analyze:

1. **Sentiment Metrics**
   - Overall sentiment distribution (positive, negative, neutral)
   - Sentiment variance (high variance = polarization signal)
   - Rapid sentiment shifts

2. **Engagement Patterns**
   - Retweet/like velocity
   - Reply-to-tweet ratio (high replies = engagement/debate)
   - User diversity (few users = echo chamber, many = broad interest)

3. **Linguistic Features**
   - Emotional intensity words
   - Controversial keywords
   - Question marks and exclamation points (indicating debate)
   - Emoji usage patterns

4. **Network Signals**
   - User interaction patterns
   - Influencer involvement
   - Echo chamber detection

5. **Temporal Patterns**
   - Tweet frequency acceleration
   - Time gaps between tweets

### Model Options
- **Baseline**: Logistic Regression (simple, interpretable)
- **Advanced**: Random Forest or XGBoost (better accuracy)
- **Stretch**: LSTM/Transformer for sequential patterns

---

## 📅 24-Hour Development Timeline

### **Hour 0-2: Setup & Data Collection**
**Goal**: Environment ready, data sourced

- [ ] Set up Python environment with all dependencies
- [ ] Decide on N (20, 50, or 100 tweets) based on data availability
- [ ] Source data:
  - **Option A**: Use Twitter API (requires API keys)
  - **Option B**: Use pre-collected datasets (Kaggle, academic datasets)
  - **Option C**: Create synthetic data for proof-of-concept
- [ ] Label some topics manually (polarized vs. died off)
  - Examples: 
    - **Polarized**: Political topics, controversial news
    - **Died Off**: Minor celebrity gossip, random trends

**Deliverable**: `data/` folder with raw tweet data

---

### **Hour 2-6: Data Exploration & Preprocessing**
**Goal**: Clean data, understand patterns

- [ ] Load tweets into pandas DataFrame
- [ ] Clean text (remove URLs, mentions, special characters)
- [ ] Exploratory Data Analysis (EDA):
  - Plot sentiment distributions
  - Analyze engagement metrics
  - Look for patterns distinguishing polarized vs. died-off topics
- [ ] Create labeled dataset with ground truth
- [ ] Split data: 70% train, 15% validation, 15% test

**Deliverable**: `notebooks/EDA.ipynb` with insights

---

### **Hour 6-12: Feature Engineering & Model Development**
**Goal**: Build predictive features and first model

- [ ] Implement feature extraction pipeline:
  - Sentiment analysis (VADER or TextBlob)
  - Engagement metrics calculation
  - Text statistics (length, caps usage, etc.)
- [ ] Create feature matrix for ML
- [ ] Train baseline model (Logistic Regression)
- [ ] Evaluate performance (accuracy, precision, recall, F1)
- [ ] Feature importance analysis
- [ ] Iterate: add/remove features to improve model

**Deliverable**: `src/feature_engineering.py`, `src/model.py`

---

### **Hour 12-16: Model Optimization & Validation**
**Goal**: Improve accuracy, validate robustness

- [ ] Try advanced models (Random Forest, XGBoost)
- [ ] Hyperparameter tuning
- [ ] Cross-validation to prevent overfitting
- [ ] Test on held-out test set
- [ ] Error analysis: which topics are hard to predict?
- [ ] Document model performance metrics

**Deliverable**: `models/` folder with trained models, performance report

---

### **Hour 16-20: Visualization & Demo Creation**
**Goal**: Make results presentable and impressive

- [ ] Create visualization showing:
  - Model predictions vs. actual outcomes
  - Feature importance
  - Example tweets from polarized vs. died-off topics
- [ ] Build simple demo interface:
  - **Option A**: Streamlit web app (fastest)
  - **Option B**: Flask/FastAPI backend + simple HTML frontend
  - **Option C**: Jupyter notebook with interactive widgets
- [ ] Add "prediction confidence" scores
- [ ] Include 2-3 case studies with explanations

**Deliverable**: `app.py` or demo notebook, visualizations in `data/visualizations/`

---

### **Hour 20-23: Polish & Presentation Prep**
**Goal**: Finalize everything for judging

- [ ] Write clear README with:
  - Project description
  - How to run the demo
  - Results summary
- [ ] Create slide deck or demo script
- [ ] Prepare 3-minute pitch highlighting:
  - The problem (predicting polarization early)
  - Your approach (features + ML)
  - Results (accuracy, cool examples)
  - Impact/novelty
- [ ] Test demo end-to-end
- [ ] Clean up code and add comments
- [ ] Push everything to GitHub

**Deliverable**: Polished demo, presentation materials

---

### **Hour 23-24: Buffer & Contingency**
**Goal**: Handle unexpected issues, practice presentation

- [ ] Fix any last-minute bugs
- [ ] Practice presenting together
- [ ] Prepare for Q&A (what if judges ask about accuracy, datasets, etc.)
- [ ] Celebrate! 🎉

---

## 🛠️ Tech Stack

### Core Libraries
- **Data Processing**: `pandas`, `numpy`
- **NLP**: `nltk`, `textblob` or `vaderSentiment`
- **Machine Learning**: `scikit-learn`, optionally `xgboost`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **API (if using Twitter)**: `tweepy`

### Demo Options
- **Quick**: Jupyter notebook with widgets
- **Web App**: `streamlit` (easiest) or `flask`
- **Advanced**: React frontend (if you have time)

---

## 📦 Project Structure

```
Durhack-2025/
├── data/
│   ├── raw/                 # Original tweet data
│   ├── processed/           # Cleaned and labeled data
│   └── visualizations/      # Charts and graphs
├── notebooks/
│   ├── EDA.ipynb           # Exploratory analysis
│   └── experiments.ipynb    # Model experiments
├── src/
│   ├── data_collection.py   # Tweet gathering
│   ├── preprocessing.py     # Data cleaning
│   ├── feature_engineering.py  # Extract features
│   ├── model.py            # ML model training
│   └── predict.py          # Make predictions
├── models/
│   └── trained_model.pkl    # Saved model
├── app.py                   # Demo application
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md         # This file!
```

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)
- [ ] Can predict polarization for at least 10 test topics
- [ ] Achieves >70% accuracy
- [ ] Has a working demo (even if just a notebook)
- [ ] Can explain the approach clearly

### Stretch Goals
- [ ] >80% accuracy
- [ ] Interactive web demo
- [ ] Real-time Twitter integration
- [ ] Published on GitHub with documentation

---

## 📝 Tips for Success

1. **Start Simple**: Get a basic working model first, then improve
2. **Time-box**: Don't spend >2 hours stuck on one problem
3. **Document as You Go**: Write down insights immediately
4. **Test Frequently**: Make sure each component works before moving on
5. **Prepare Demo Early**: Start at Hour 16, not Hour 23!
6. **Divide & Conquer**: Split tasks between team members
7. **Use Pre-trained Models**: Don't train sentiment analysis from scratch
8. **Focus on Story**: Judges care about novelty and presentation, not just accuracy

---

## 🚀 Quick Start Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run exploratory analysis
jupyter notebook notebooks/EDA.ipynb

# Run main script
python main.py

# Launch demo (if using Streamlit)
streamlit run app.py
```

---

## 🤝 Team Workflow Suggestions

### Pair Programming Approach
- **Person A**: Data collection & preprocessing
- **Person B**: Feature engineering & modeling
- **Both**: Alternate on demo/visualization

### Checkpoint Times
- **Hour 6**: Review EDA findings together
- **Hour 12**: Review first model results
- **Hour 18**: Review demo progress
- **Hour 22**: Final review and practice pitch

---

## 📚 Useful Resources

### Datasets
- [Twitter Sentiment Analysis Dataset (Kaggle)](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- [Controversial Topics Dataset](https://www.kaggle.com/datasets)
- Academic papers on polarization detection

### Tools
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) - Great for social media
- [Streamlit Gallery](https://streamlit.io/gallery) - Demo inspiration
- [Scikit-learn Docs](https://scikit-learn.org/) - ML reference

---

## 🏆 Presentation Tips

### What Judges Want to See
1. **Novelty**: How is this different/interesting?
2. **Technical Depth**: Real ML, not just an API wrapper
3. **Polish**: Working demo beats perfect code
4. **Clarity**: Can you explain it in 3 minutes?
5. **Impact**: Why does this matter?

### Pitch Structure (3 minutes)
1. **The Hook** (30s): "Did you know most controversial topics show warning signs in the first 20 tweets?"
2. **The Problem** (30s): "We predict if topics will polarize or die off early"
3. **The Solution** (60s): Show demo with real example
4. **The Results** (45s): Accuracy, insights, cool findings
5. **The Impact** (15s): Real-world applications

---

## ✅ Pre-Submission Checklist

- [ ] Code runs without errors
- [ ] README is clear and complete
- [ ] Demo works end-to-end
- [ ] GitHub repo is public and organized
- [ ] Presentation slides/script ready
- [ ] All team members understand the project
- [ ] Tested on fresh environment (no "works on my machine")

---

Good luck! You've got this! 🚀

Remember: **Done is better than perfect** in a hackathon. Ship something working, then iterate!
