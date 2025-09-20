from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42)
        }
        self.trained_models = {}
        self.feature_importance = {}

    def train_models(self, X_train, y_train):
        """Train all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        for name, model in self.trained_models.items():
            predictions = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, predictions),
                'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                'confusion_matrix': confusion_matrix(y_test, predictions),
                'classification_report': classification_report(y_test, predictions)
            }
        return results

    def get_feature_importance(self, feature_names):
        """Get feature importance for each model"""
        for name, model in self.trained_models.items():
            if name == 'logistic':
                importance = model.coef_[0]
            elif name in ['random_forest', 'xgboost']:
                importance = model.feature_importances_
            
            self.feature_importance[name] = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return self.feature_importance

    def predict(self, X, model_name='xgboost'):
        """Make predictions using a specific model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        return self.trained_models[model_name].predict(X)

    def predict_proba(self, X, model_name='xgboost'):
        """Get prediction probabilities using a specific model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        return self.trained_models[model_name].predict_proba(X)