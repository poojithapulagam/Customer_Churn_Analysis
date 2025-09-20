import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load and preprocess the customer churn data"""
        # Read the CSV file
        self.data = pd.read_csv(self.file_path)
        
        # Convert TotalCharges to numeric, handling any errors
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
        
        # Drop rows with missing values
        self.data = self.data.dropna()
        
        # Remove customer ID column if it exists
        if 'customerID' in self.data.columns:
            self.data = self.data.drop('customerID', axis=1)
            
        return self.data

    def prepare_features(self):
        """Prepare features and target variable"""
        # Separate features and target
        self.y = self.data['Churn'].map({'Yes': 1, 'No': 0})
        self.X = self.data.drop('Churn', axis=1)
        
        # Get categorical and numeric columns
        self.categorical_cols = self.X.select_dtypes(include=['object']).columns
        self.numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        
        return self.X, self.y

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_names(self):
        """Get categorical and numeric feature names"""
        return {
            'categorical': list(self.categorical_cols),
            'numeric': list(self.numeric_cols)
        }