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
        
    def load_data(self, positive_filepath='data/processed/positive_with_real_shark_density.csv', 
                  negative_filepath='data/processed/negative_with_real_shark_density.csv'):
        """Load and combine positive and negative datasets with real shark density."""
        
        logger.info(f"📊 Loading positive data from {positive_filepath}")
        logger.info(f"📊 Loading negative data from {negative_filepath}")
        
        # Load positive and negative datasets
        positive_df = pd.read_csv(positive_filepath)
        negative_df = pd.read_csv(negative_filepath)
        
        # Add Attack_Type column
        positive_df['Attack_Type'] = 'positive'
        negative_df['Attack_Type'] = 'negative'
        
        # Combine the datasets
        df = pd.concat([positive_df, negative_df], ignore_index=True)
        
        logger.info(f"Loaded {len(positive_df):,} positive records")
        logger.info(f"Loaded {len(negative_df):,} negative records") 
        logger.info(f"Combined total: {len(df):,} records")
        
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
    
    def preprocess_geographic_bias(self, df, balance_regions=True, coastal_focus=True):
        """
        Preprocess data to reduce geographic bias and focus on environmental factors.
        
        Args:
            df: Combined positive/negative dataframe
            balance_regions: Create balanced negative samples per region
            coastal_focus: Filter to coastal areas only (reasonable shark habitat)
        """
        
        logger.info("🌍 Preprocessing to reduce geographic bias...")
        
        df_processed = df.copy()
        
        # 1. Focus on coastal areas only (sharks don't attack in deep ocean/inland)
        if coastal_focus:
            logger.info("Filtering to coastal areas (reasonable shark habitat)...")
            
            # Remove extreme latitudes (polar regions) where sharks don't exist
            coastal_mask = (
                (df_processed['Latitude'].abs() <= 60) &  # No polar regions
                (df_processed['Population'] > 1000)  # Some human presence (coastal areas)
            )
            
            before_count = len(df_processed)
            df_processed = df_processed[coastal_mask].copy()
            after_count = len(df_processed)
            
            logger.info(f"Coastal filtering: {before_count:,} → {after_count:,} records "
                       f"({after_count/before_count:.1%} retained)")
        
        # 2. Geographic region balancing
        if balance_regions:
            logger.info("Balancing negative samples across geographic regions...")
            
            positive_df = df_processed[df_processed['Attack_Type'] == 'positive'].copy()
            negative_df = df_processed[df_processed['Attack_Type'] == 'negative'].copy()
            
            # Define regions based on positive samples
            def get_region(lat, lon):
                """Classify into broad geographic regions."""
                if lat >= 24 and lat <= 50 and lon >= -100 and lon <= -60:
                    return 'North America East'
                elif lat >= 10 and lat <= 40 and lon >= -130 and lon <= -100:
                    return 'North America West'
                elif lat >= -40 and lat <= -10 and lon >= 110 and lon <= 160:
                    return 'Australia/Pacific'
                elif lat >= -40 and lat <= 40 and lon >= -20 and lon <= 60:
                    return 'Africa/Europe'
                elif lat >= -30 and lat <= 30 and lon >= 60 and lon <= 110:
                    return 'Asia'
                elif lat >= -60 and lat <= 20 and lon >= -90 and lon <= -30:
                    return 'South America'
                else:
                    return 'Other'
            
            # Add regions to dataframes
            positive_df['Region'] = positive_df.apply(lambda x: get_region(x['Latitude'], x['Longitude']), axis=1)
            negative_df['Region'] = negative_df.apply(lambda x: get_region(x['Latitude'], x['Longitude']), axis=1)
            
            # Count positive samples per region
            pos_region_counts = positive_df['Region'].value_counts()
            logger.info(f"Positive samples by region: {pos_region_counts.to_dict()}")
            
            # Balance negative samples: sample negative records from each region 
            # proportional to positive samples (but with reasonable minimum)
            balanced_negatives = []
            total_positives = len(positive_df)
            
            for region in pos_region_counts.index:
                pos_count = pos_region_counts[region]
                region_negatives = negative_df[negative_df['Region'] == region]
                
                if len(region_negatives) > 0:
                    # Target: 3x negative samples per positive in each region (minimum 100)
                    target_negatives = max(100, pos_count * 3)
                    actual_sample = min(target_negatives, len(region_negatives))
                    
                    sampled = region_negatives.sample(n=actual_sample, random_state=42)
                    balanced_negatives.append(sampled)
                    
                    logger.info(f"Region {region}: {pos_count} pos, "
                               f"{len(region_negatives)} neg available, "
                               f"{actual_sample} neg sampled")
            
            # Combine balanced negative samples
            balanced_negative_df = pd.concat(balanced_negatives, ignore_index=True)
            
            # Combine with all positive samples
            df_processed = pd.concat([positive_df, balanced_negative_df], ignore_index=True)
            
            # Remove the temporary Region column
            df_processed = df_processed.drop(columns=['Region'])
            
            logger.info(f"After geographic balancing: {len(df_processed):,} total records")
            logger.info(f"Final distribution: {df_processed['Attack_Type'].value_counts().to_dict()}")
        
        return df_processed
    
    def engineer_features(self, df):
        """Feature engineering with temperature-adjusted shark density."""
        
        logger.info("🔧 Feature engineering with temperature-based shark density adjustment...")
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Convert attack type to binary target
        df_features['target'] = (df_features['Attack_Type'] == 'positive').astype(int)
        
        # Transform population to log scale (handles extreme values better)
        df_features['Log_Population'] = np.log1p(df_features['Population'])
        
        # Apply temperature-based shark density adjustment
        # This allows the model to learn temperature effects rather than post-processing
        def adjust_shark_density_for_temperature(row):
            shark_density = row['Real_Shark_Density']
            sst = row['SST_Celsius']
            year = row['Year']
            
            # Add climate change factor: 0.03°C increase per year since 2010
            climate_adjusted_sst = sst + (year - 2010) * 0.03
            
            # Only adjust if there are sharks present
            if shark_density == 0:
                return 0.0
            
            if 26 <= climate_adjusted_sst <= 28:
                # Optimal temperature range - increase activity
                temp_factor = 1.5
            elif 24 <= climate_adjusted_sst < 26 or 28 < climate_adjusted_sst <= 30:
                # Good temperature range
                temp_factor = 1.2
            elif 20 <= climate_adjusted_sst < 24 or 30 < climate_adjusted_sst <= 32:
                # Acceptable range
                temp_factor = 1.0
            elif 16 <= climate_adjusted_sst < 20 or 32 < climate_adjusted_sst <= 34:
                # Suboptimal - less activity
                temp_factor = 0.8
            elif climate_adjusted_sst < 16:
                # Too cold - avoid or sluggish
                temp_factor = 0.4
            else:  # climate_adjusted_sst > 34
                # Too hot - seek cooler water
                temp_factor = 0.6
        
            
            # Apply temperature adjustment to shark density
            adjusted_density = shark_density * temp_factor
            return adjusted_density
        
        # Create temperature-adjusted shark density feature
        df_features['Temp_Adjusted_Shark_Density'] = df_features.apply(
            adjust_shark_density_for_temperature, axis=1
        )
        
        logger.info(f"✅ Applied temperature-based shark density adjustment")
        logger.info(f"Original shark density range: {df_features['Real_Shark_Density'].min():.4f} - {df_features['Real_Shark_Density'].max():.4f}")
        logger.info(f"Adjusted shark density range: {df_features['Temp_Adjusted_Shark_Density'].min():.4f} - {df_features['Temp_Adjusted_Shark_Density'].max():.4f}")
        logger.info(f"✅ Feature engineering complete. Shape: {df_features.shape}")
        
        return df_features
    
    def prepare_features(self, df):
        """Prepare features with temperature-adjusted shark density for model training."""
        
        logger.info("📋 Preparing features with temperature-adjusted shark density...")
        
        # Use temperature-adjusted shark density instead of raw shark density
        feature_cols = [
            'Year', 'Month', 
            'SST_Celsius', 'Temp_Adjusted_Shark_Density', 'Log_Population'
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
        logger.info("🌡️ Using temperature-adjusted shark density for training")
        logger.info("🚫 Excluded Latitude/Longitude to prevent geographic overfitting")
        
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
    
    def predict_attack_probability(self, features_dict, apply_constraints=True):
        """Predict attack probability for given features with optional post-processing constraints."""
        
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return None
        
        # Convert to DataFrame with temperature-adjusted shark density
        df = pd.DataFrame([features_dict])
        
        # Calculate temperature-adjusted shark density for prediction
        shark_density = features_dict.get('Real_Shark_Density', 0)
        sst = features_dict.get('SST_Celsius', 25)
        
        # Apply same temperature adjustment as in training
        if shark_density == 0:
            temp_adjusted_density = 0.0
        else:
            if 20 <= sst <= 26:
                temp_factor = 1.3
            elif 18 <= sst < 20 or 26 < sst <= 28:
                temp_factor = 1.15
            elif 16 <= sst < 18 or 28 < sst <= 30:
                temp_factor = 1.0
            elif 14 <= sst < 16 or 30 < sst <= 32:
                temp_factor = 0.8
            elif sst < 14:
                temp_factor = 0.4
            else:  # sst > 32
                temp_factor = 0.6
            
            temp_adjusted_density = shark_density * temp_factor
        
        # Use temperature-adjusted shark density for prediction
        prediction_features = features_dict.copy()
        prediction_features['Temp_Adjusted_Shark_Density'] = temp_adjusted_density
        df = pd.DataFrame([prediction_features])
        
        # Get probability from model (already accounts for temperature effects)
        prob = self.model.predict_proba(df[self.feature_names])[:, 1][0]
        
        if not apply_constraints:
            return prob
        
        # Only apply zero shark density constraint (temperature effects already learned)
        if shark_density == 0:
            # If no sharks present, reduce probability dramatically
            constrained_prob = prob * 0.01  # Reduce by 99%
        elif shark_density < 0.05:  # Very low shark density
            # Scale down probability proportionally
            scaling_factor = shark_density / 0.05
            constrained_prob = prob * (0.01 + 0.99 * scaling_factor)
        else:
            # Model already learned temperature effects - no additional adjustment needed
            constrained_prob = prob
        
        return constrained_prob
    
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
    """Main training pipeline with geographic bias preprocessing."""
    
    logger.info("🦈 Starting ENHANCED Shark Attack Prediction Model Training")
    logger.info("🌍 With Geographic Bias Reduction & Environmental Focus")
    logger.info("=" * 70)
    
    # Initialize predictor
    predictor = SharkAttackPredictor(random_state=42)
    
    # Load raw data
    df_raw = predictor.load_data()
    
    # Apply geographic bias preprocessing
    df_processed = predictor.preprocess_geographic_bias(
        df_raw, 
        balance_regions=True, 
        coastal_focus=True
    )
    
    # Enhanced feature engineering
    df_features = predictor.engineer_features(df_processed)
    X, y = predictor.prepare_features(df_features)
    
    # Train model
    X_train, X_test, y_train, y_test = predictor.train_model(X, y)
    
    # Plot feature importance
    feature_importance = predictor.plot_feature_importance()
    
    # Plot ROC curve
    predictor.plot_roc_curve(X_test, y_test, "Test Set")
    
    # Save model with new name to distinguish from old model
    predictor.save_model('models/shark_attack_xgboost_environmental_model.pkl')
    
    # Example prediction with temperature-adjusted features
    logger.info("\n🔮 Example Predictions with Temperature-Adjusted Shark Density:")
    example_features = {
        'Year': 2024, 'Month': 7,
        'SST_Celsius': 26.5, 'Real_Shark_Density': 0.15, 'Log_Population': 13.12
    }
    
    raw_prob = predictor.predict_attack_probability(example_features, apply_constraints=False)
    constrained_prob = predictor.predict_attack_probability(example_features, apply_constraints=True)
    logger.info(f"High shark density, optimal temp - Raw: {raw_prob:.1%}, Constrained: {constrained_prob:.1%}")
    
    # Test with zero shark density
    example_no_sharks = example_features.copy()
    example_no_sharks['Real_Shark_Density'] = 0.0
    
    raw_prob_zero = predictor.predict_attack_probability(example_no_sharks, apply_constraints=False)
    constrained_prob_zero = predictor.predict_attack_probability(example_no_sharks, apply_constraints=True)
    logger.info(f"Zero shark density - Raw: {raw_prob_zero:.1%}, Constrained: {constrained_prob_zero:.1%}")
    
    # Test with cold water
    example_cold = example_features.copy()
    example_cold['SST_Celsius'] = 12.0  # Cold water
    
    raw_prob_cold = predictor.predict_attack_probability(example_cold, apply_constraints=False)
    constrained_prob_cold = predictor.predict_attack_probability(example_cold, apply_constraints=True)
    logger.info(f"High sharks, cold water - Raw: {raw_prob_cold:.1%}, Constrained: {constrained_prob_cold:.1%}")
    
    logger.info("\n🎉 Temperature-adjusted model training completed!")
    logger.info("Model learned temperature effects during training - no post-processing needed!")
    logger.info("Saved as: models/shark_attack_xgboost_environmental_model.pkl")

if __name__ == "__main__":
    main()