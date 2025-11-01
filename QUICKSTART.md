# Quick Reference - Durham Hackathon 2025

## 🚀 Getting Started Checklist

### Setup (Already Done! ✅)
- [x] Virtual environment activated (`.venv`)
- [x] All dependencies installed
- [x] Project structure created
- [x] Starter code modules created
- [x] Pushed to GitHub

### Your Next Steps

#### 1️⃣ **Data Collection** (Hour 0-2)
```bash
# Option A: Use pre-collected datasets
# - Download from Kaggle or academic sources
# - Look for Twitter sentiment datasets
# - Save to data/raw/

# Option B: Create sample data for testing
# - Manually create a small CSV with columns:
#   - text, retweet_count, like_count, reply_count, label
# - Label: 0 = died off, 1 = polarized
```

**Files to work with:**
- `src/data_collection.py` - Modify for your data source
- Save data to: `data/raw/tweets.csv`

#### 2️⃣ **Explore & Analyze** (Hour 2-6)
```bash
# Open Jupyter notebook
jupyter notebook notebooks/EDA.ipynb
```

**What to do:**
- Load your data
- Visualize sentiment distributions
- Look for patterns between polarized vs. died-off topics
- Try the preprocessing and feature extraction functions

**Code snippets to use:**
```python
import pandas as pd
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import create_feature_vector

# Load data
df = pd.read_csv('../data/raw/tweets.csv')

# Clean text
df = preprocess_dataframe(df)

# Extract features for one topic
features = create_feature_vector(df)
print(features)
```

#### 3️⃣ **Build Model** (Hour 6-12)
```python
from src.model import PolarizationPredictor
from sklearn.model_selection import train_test_split

# Prepare data
X = feature_matrix  # Your extracted features
y = labels  # 0 or 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = PolarizationPredictor(model_type='logistic')
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(metrics)

# Save
model.save_model('models/polarization_model.pkl')
```

#### 4️⃣ **Create Demo** (Hour 16-20)
```bash
# Option 1: Streamlit app (fastest!)
streamlit run app.py
```

Create `app.py`:
```python
import streamlit as st
import pandas as pd
from src.model import PolarizationPredictor
from src.feature_engineering import create_feature_vector

st.title("Tweet Polarization Predictor 🔮")

# Load model
model = PolarizationPredictor.load_model('models/polarization_model.pkl')

# Input
st.write("Upload first 100 tweets on a topic:")
uploaded = st.file_uploader("CSV file", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    features = create_feature_vector(df)
    prediction = model.predict([list(features.values())])
    
    if prediction[0] == 1:
        st.error("⚠️ This topic will likely POLARIZE!")
    else:
        st.success("✅ This topic will likely DIE OFF")
```

## 📊 Sample Data Structure

Your CSV should look like:
```csv
text,retweet_count,like_count,reply_count,label
"This is controversial!",50,120,80,1
"Just saw this movie",2,5,1,0
```

Where:
- `label`: 0 = died off, 1 = polarized

## 🎯 Success Tips

1. **Start with 2-3 labeled topics** (don't need 100s to start)
2. **Focus on getting something working** first
3. **Document your findings** as you go
4. **Prepare demo early** - start at Hour 16!
5. **Practice your pitch** - explain it simply

## 🔥 Cool Things to Show Judges

- "Our model achieved X% accuracy on predicting polarization"
- Live demo: "Let's predict if this topic will polarize"
- Feature importance: "Sentiment variance was the top predictor"
- Case study: "Look at this topic that polarized - we predicted it!"

## 📁 Project Structure Reference

```
Durhack-2025/
├── data/
│   ├── raw/              # Your collected tweets
│   └── processed/        # Cleaned & labeled data
├── notebooks/
│   └── EDA.ipynb        # Start here for exploration!
├── src/
│   ├── data_collection.py    # Modify for your source
│   ├── preprocessing.py      # Ready to use
│   ├── feature_engineering.py # Ready to use
│   └── model.py             # Ready to use
├── models/              # Save trained models here
├── main.py             # Test basic workflow
├── app.py              # Create this for demo
└── PROJECT_PLAN.md     # Full 24h timeline
```

## 💻 Useful Commands

```bash
# Activate environment
source .venv/bin/activate

# Run main script
python main.py

# Open Jupyter
jupyter notebook

# Run Streamlit demo
streamlit run app.py

# Git commands
git add -A
git commit -m "Your message"
git push origin main
```

## 🆘 If You Get Stuck

1. **Can't find data?** Create 10 fake topics with labels to test your pipeline
2. **Model not training?** Start with logistic regression, it's simpler
3. **No time for demo?** Use Jupyter notebook as your demo
4. **Low accuracy?** That's OK! Focus on interesting insights instead

## 🎤 3-Minute Pitch Template

1. **Hook** (30s): "Most controversial topics show warning signs in the first 20 tweets"
2. **Problem** (30s): "We predict polarization vs die-off early"
3. **Demo** (60s): [Show live prediction]
4. **Results** (30s): "X% accurate, sentiment variance is key predictor"
5. **Impact** (30s): "Helps platforms/brands detect controversy early"

---

**You're all set! Good luck at the hackathon! 🚀**

See `PROJECT_PLAN.md` for the complete 24-hour breakdown.
