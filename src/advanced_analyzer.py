import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class AdvancedAnalyzer:
    def __init__(self, data, X_processed, y):
        self.data = data
        self.X_processed = X_processed
        self.y = y
        
    def create_advanced_visualizations(self):
        """Create detailed visualizations for key relationships"""
        # Set up the visualization style
        plt.style.use('seaborn')
        
        # 1. Churn Rate by Monthly Charges Segments
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 15))
        
        # Monthly Charges Distribution
        sns.histplot(data=self.data, x='MonthlyCharges', hue='Churn', 
                    multiple="stack", ax=axes1[0,0])
        axes1[0,0].set_title('Monthly Charges Distribution by Churn Status')
        
        # Churn Rate by Monthly Charges Quintiles
        self.data['ChargesBin'] = pd.qcut(self.data['MonthlyCharges'], q=5)
        churn_by_charges = self.data.groupby('ChargesBin')['Churn'].apply(
            lambda x: (x == 'Yes').mean()
        )
        churn_by_charges.plot(kind='bar', ax=axes1[0,1])
        axes1[0,1].set_title('Churn Rate by Monthly Charges Quintiles')
        
        # Multiple Services Impact
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport']
        self.data['NumberOfServices'] = self.data[service_cols].apply(
            lambda x: sum(x != 'No'), axis=1
        )
        sns.boxplot(data=self.data, x='NumberOfServices', y='MonthlyCharges', 
                   hue='Churn', ax=axes1[1,0])
        axes1[1,0].set_title('Monthly Charges by Number of Services')
        
        # Churn Rate by Contract and Payment Method
        contract_payment_churn = pd.crosstab(
            [self.data['Contract'], self.data['PaymentMethod']],
            self.data['Churn'],
            normalize='index'
        )['Yes']
        contract_payment_churn.unstack().plot(kind='bar', ax=axes1[1,1])
        axes1[1,1].set_title('Churn Rate by Contract and Payment Method')
        
        plt.tight_layout()
        
        # 2. Customer Segments Analysis
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 15))
        
        # Tenure Distribution
        sns.histplot(data=self.data, x='tenure', hue='Churn', 
                    multiple="stack", ax=axes2[0,0])
        axes2[0,0].set_title('Customer Tenure Distribution by Churn Status')
        
        # Average Monthly Charges by Tenure
        self.data['TenureBin'] = pd.qcut(self.data['tenure'], q=5)
        sns.boxplot(data=self.data, x='TenureBin', y='MonthlyCharges', 
                   hue='Churn', ax=axes2[0,1])
        axes2[0,1].set_title('Monthly Charges by Tenure Quintiles')
        
        # Service Adoption Patterns
        service_adoption = self.data[service_cols].apply(
            lambda x: (x != 'No').mean()
        )
        service_adoption.plot(kind='bar', ax=axes2[1,0])
        axes2[1,0].set_title('Service Adoption Rates')
        
        # Customer Value Segments
        self.data['CustomerValue'] = self.data['MonthlyCharges'] * self.data['tenure']
        sns.scatterplot(data=self.data, x='tenure', y='MonthlyCharges', 
                       hue='Churn', size='CustomerValue', sizes=(20, 200),
                       alpha=0.6, ax=axes2[1,1])
        axes2[1,1].set_title('Customer Value Segments')
        
        plt.tight_layout()
        
        return [fig1, fig2]
    
    def perform_advanced_modeling(self):
        """Implement SVM with hyperparameter tuning"""
        # Scale the features for SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_processed)
        
        # Define SVM parameters for grid search
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.1]
        }
        
        # Perform grid search
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_scaled, self.y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'model': grid_search.best_estimator_
        }
    
    def calculate_roi_impact(self):
        """Calculate detailed ROI for recommendations"""
        avg_monthly_revenue = self.data['MonthlyCharges'].mean()
        avg_customer_lifetime = self.data['tenure'].mean()
        total_customers = len(self.data)
        current_churn_rate = (self.data['Churn'] == 'Yes').mean()
        
        # Calculate current state
        yearly_lost_customers = total_customers * current_churn_rate
        yearly_lost_revenue = yearly_lost_customers * avg_monthly_revenue * 12
        
        # Calculate impact of recommendations
        recommendations_impact = {
            'contract_strategy': {
                'description': 'Implement contract incentives',
                'implementation_cost': 50000,  # Example cost
                'churn_reduction': 0.2,  # 20% reduction in month-to-month segment
                'affected_customers': len(self.data[self.data['Contract'] == 'Month-to-month']),
                'yearly_savings': 0
            },
            'service_quality': {
                'description': 'Improve fiber optic service',
                'implementation_cost': 150000,  # Example cost
                'churn_reduction': 0.3,  # 30% reduction in fiber optic segment
                'affected_customers': len(self.data[self.data['InternetService'] == 'Fiber optic']),
                'yearly_savings': 0
            },
            'payment_methods': {
                'description': 'Promote automatic payments',
                'implementation_cost': 25000,  # Example cost
                'churn_reduction': 0.25,  # 25% reduction in electronic check segment
                'affected_customers': len(self.data[self.data['PaymentMethod'] == 'Electronic check']),
                'yearly_savings': 0
            }
        }
        
        # Calculate ROI for each recommendation
        for rec in recommendations_impact.values():
            potential_saved_customers = rec['affected_customers'] * current_churn_rate * rec['churn_reduction']
            rec['yearly_savings'] = potential_saved_customers * avg_monthly_revenue * 12
            rec['roi'] = (rec['yearly_savings'] - rec['implementation_cost']) / rec['implementation_cost']
            rec['payback_period'] = rec['implementation_cost'] / rec['yearly_savings'] * 12  # in months
        
        return recommendations_impact