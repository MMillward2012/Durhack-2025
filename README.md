# Tweet Polarization Predictor 🔮
**Durham Hackathon 2025 - Predicting the Future Challenge**

## 🎯 What We're Building
Can we predict whether a topic will **polarize** or **die off** based on just the first 20-100 tweets? 

This project uses machine learning to analyze early signals in Twitter conversations and forecast whether a topic will:
- **Polarize**: Generate heated debate and strong opposing viewpoints
- **Die Off**: Fade away without gaining significant traction

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
├── data/              # Tweet datasets (raw & processed)
├── notebooks/         # Jupyter notebooks for EDA
├── src/               # Source code (preprocessing, models, features)
├── models/            # Trained ML models
├── tests/             # Unit tests
├── PROJECT_PLAN.md    # Detailed 24-hour hackathon plan
└── requirements.txt   # Python dependencies
```

## 📖 Full Project Plan
**→ See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete 24-hour development timeline, technical approach, and presentation tips!**

## 🛠️ Tech Stack
- **Data**: pandas, numpy
- **NLP**: nltk, vaderSentiment, textblob
- **ML**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn, plotly
- **Demo**: streamlit (stretch goal)

## 👥 Team
Durham Hackathon 2025 participants working on the "Predict the Future" challenge!

## 📝 License
See LICENSE file for details.