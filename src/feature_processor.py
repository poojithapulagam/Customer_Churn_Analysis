import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeatureProcessor:
    def __init__(self, categorical_features, numeric_features):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.preprocessor = None

    def create_preprocessor(self):
        """Create preprocessing pipeline for both numeric and categorical features"""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.preprocessor

    def fit_transform(self, X_train):
        """Fit and transform the training data"""
        if self.preprocessor is None:
            self.create_preprocessor()
        return self.preprocessor.fit_transform(X_train)

    def transform(self, X_test):
        """Transform the test data"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        return self.preprocessor.transform(X_test)

    def get_feature_names(self):
        """Get the names of transformed features"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
        numeric_features = self.numeric_features
        categorical_features_encoded = []
        
        for i, feature in enumerate(self.categorical_features):
            encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            categories = encoder.categories_[i][1:]  # Skip first category (dropped)
            categorical_features_encoded.extend([f"{feature}_{cat}" for cat in categories])
            
        return numeric_features + categorical_features_encoded