"""
Machine learning model for polarization prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import pickle


class PolarizationPredictor:
    """
    Model to predict if a topic will polarize or die off.
    """
    
    def __init__(self, model_type='logistic'):
        """
        Initialize the predictor.
        
        Args:
            model_type: 'logistic' or 'random_forest'
        """
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")
        
        self.feature_names = None
    
    def train(self, X, y):
        """
        Train the model.
        
        Args:
            X: Feature matrix (DataFrame or numpy array)
            y: Target labels (0=died off, 1=polarized)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (0 or 1)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability for each class
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance (for tree-based models).
        
        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            PolarizationPredictor object
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
