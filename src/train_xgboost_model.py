#!/usr/bin/env python3
"""
XGBoost Shark Attack Prediction Model
====================================

Trains an XGBoost classifier on shark attack data to predict attack probability
based on environmental factors (SST, population, shark density) and temporal features.

Features:
- Real shark density from GBIF data
- Sea surface temperature 
- Human population density
- Temporal features (month, season)
- Geographic features

Author: GitHub Copilot
Date: November 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, roc_curve,
                            accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SharkAttackPredictor:
    """XGBoost-based shark attack prediction model."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def load_data(self, filepath='data/processed/final_real_shark_data.csv'):
        """Load and prepare the shark attack dataset."""
        
        logger.info(f"📊 Loading data from {filepath}")
        
        # Load the data
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} records")
        
        # Basic data info
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Data shape: {df.shape}")
        
        # Check target distribution
        target_counts = df['Attack_Type'].value_counts()
        logger.info(f"Target distribution:")
        for attack_type, count in target_counts.items():
            percentage = count / len(df) * 100
            logger.info(f"  {attack_type}: {count:,} ({percentage:.1f}%)")
        
        return df
    
    def engineer_features(self, df):
        """Minimal feature engineering - just convert target and log population."""
        
        logger.info("🔧 Minimal feature engineering...")
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Convert attack type to binary target
        df_features['target'] = (df_features['Attack_Type'] == 'positive').astype(int)
        
        # Only transform population to log scale (handles extreme values better)
        df_features['Log_Population'] = np.log1p(df_features['Population'])
        
        logger.info(f"✅ Feature engineering complete. Shape: {df_features.shape}")
        
        return df_features
    
    def prepare_features(self, df):
        """Prepare features for model training."""
        
        logger.info("📋 Preparing features for training...")
        
        # Use original measurements + log population
        feature_cols = [
            'Year', 'Month', 'Latitude', 'Longitude', 
            'SST_Celsius', 'Real_Shark_Density', 'Log_Population'
        ]
        
        # Handle any missing features gracefully
        available_features = [col for col in feature_cols if col in df.columns]
        if len(available_features) != len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        self.feature_names = available_features
        
        X = df[available_features].copy()
        y = df['target'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        logger.info(f"✅ Prepared {len(available_features)} features for {len(X):,} samples")
        logger.info(f"Features: {available_features}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        """Train the XGBoost model."""
        
        logger.info("🚀 Training XGBoost model...")
        
        # Calculate class weights for imbalanced data
        n_positive = (y == 1).sum()
        n_negative = (y == 0).sum()
        scale_pos_weight = n_negative / n_positive
        
        logger.info(f"Class distribution: {n_positive:,} positive, {n_negative:,} negative")
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Initialize XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Train with early stopping
        logger.info("Training with early stopping...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        self.is_trained = True
        
        # Evaluate on test set
        self.evaluate_model(X_test, y_test, dataset_name="Test Set")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, X, y, dataset_name="Dataset"):
        """Evaluate model performance."""
        
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return
        
        logger.info(f"📈 Evaluating model on {dataset_name}...")
        
        # Get predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc_roc = roc_auc_score(y, y_pred_proba)
        
        logger.info(f"📊 {dataset_name} Results:")
        logger.info(f"  Accuracy:  {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall:    {recall:.3f}")
        logger.info(f"  F1-Score:  {f1:.3f}")
        logger.info(f"  ROC-AUC:   {auc_roc:.3f}")
        
        # Detailed classification report
        logger.info(f"\n📋 Detailed Classification Report:")
        print(classification_report(y, y_pred, target_names=['No Attack', 'Attack']))
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance."""
        
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features for Shark Attack Prediction')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('models')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        logger.info(f"🎯 Top {top_n} Most Important Features:")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            logger.info(f"  {i:2d}. {row['feature']:25s} {row['importance']:.3f}")
        
        return feature_importance
    
    def plot_roc_curve(self, X, y, dataset_name="Dataset"):
        """Plot ROC curve."""
        
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc_score = roc_auc_score(y, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('models')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'roc_curve_{dataset_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_attack_probability(self, features_dict):
        """Predict attack probability for given features."""
        
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Get probability
        prob = self.model.predict_proba(df)[:, 1][0]
        
        return prob
    
    def save_model(self, filepath='models/shark_attack_xgboost_model.pkl'):
        """Save the trained model."""
        
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return
        
        output_dir = Path(filepath).parent
        output_dir.mkdir(exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='models/shark_attack_xgboost_model.pkl'):
        """Load a trained model."""
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        logger.info(f"✅ Model loaded from {filepath}")

def main():
    """Main training pipeline."""
    
    logger.info("🦈 Starting Shark Attack Prediction Model Training")
    logger.info("=" * 60)
    
    # Initialize predictor
    predictor = SharkAttackPredictor(random_state=42)
    
    # Load and prepare data
    df = predictor.load_data()
    df_features = predictor.engineer_features(df)
    X, y = predictor.prepare_features(df_features)
    
    # Train model
    X_train, X_test, y_train, y_test = predictor.train_model(X, y)
    
    # Plot feature importance
    feature_importance = predictor.plot_feature_importance()
    
    # Plot ROC curve
    predictor.plot_roc_curve(X_test, y_test, "Test Set")
    
    # Save model
    predictor.save_model()
    
    # Example prediction
    logger.info("\n🔮 Example Prediction:")
    example_features = {
        'Year': 2024, 'Month': 7, 'Latitude': 27.0, 'Longitude': -80.0,
        'SST_Celsius': 26.5, 'Real_Shark_Density': 0.15, 'Log_Population': 13.12
    }
    
    prob = predictor.predict_attack_probability(example_features)
    logger.info(f"Attack probability for Florida coast in July: {prob:.1%}")
    
    logger.info("\n🎉 Model training completed successfully!")
    logger.info("Model saved and ready for predictions.")

if __name__ == "__main__":
    main()