"""
Main entry point for the Tweet Polarization Predictor.

This script demonstrates the basic workflow:
1. Load/collect tweet data
2. Preprocess tweets
3. Extract features
4. Train/load model
5. Make predictions
"""

import pandas as pd
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import create_feature_vector
from src.model import PolarizationPredictor


def main():
    print("=" * 60)
    print("Tweet Polarization Predictor")
    print("Durham Hackathon 2025 - Predicting the Future Challenge")
    print("=" * 60)
    print()
    
    print("Welcome to the Tweet Polarization Predictor!")
    print()
    print("This tool predicts whether a topic will:")
    print("  - POLARIZE: Generate heated debate with opposing views")
    print("  - DIE OFF: Fade away without gaining traction")
    print()
    print("Next steps:")
    print("  1. Collect or load tweet data (see src/data_collection.py)")
    print("  2. Run exploratory analysis (notebooks/EDA.ipynb)")
    print("  3. Train your model using the workflow above")
    print("  4. See PROJECT_PLAN.md for the full 24-hour timeline!")
    print()
    print("=" * 60)
    
    # Example workflow (uncomment when you have data):
    # 
    # # Load sample data
    # df = pd.read_csv('data/sample_tweets.csv')
    # 
    # # Preprocess
    # df = preprocess_dataframe(df)
    # 
    # # Extract features
    # features = create_feature_vector(df)
    # print(f"Extracted {len(features)} features")
    # 
    # # Train model (when you have labeled data)
    # # model = PolarizationPredictor(model_type='logistic')
    # # model.train(X_train, y_train)
    # # model.save_model('models/polarization_model.pkl')


if __name__ == "__main__":
    main()
