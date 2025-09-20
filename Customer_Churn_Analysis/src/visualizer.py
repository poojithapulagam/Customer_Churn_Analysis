import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, data):
        self.data = data
        self.plt = plt
        self.sns = sns
        
    def plot_feature_importance(self, feature_importance_dict):
        """Plot feature importance for different models"""
        fig, axes = plt.subplots(len(feature_importance_dict), 1, figsize=(12, 5*len(feature_importance_dict)))
        
        for i, (model_name, importance_df) in enumerate(feature_importance_dict.items()):
            ax = axes[i] if len(feature_importance_dict) > 1 else axes
            
            sns.barplot(data=importance_df.head(10), x='importance', y='feature', ax=ax)
            ax.set_title(f'Top 10 Important Features - {model_name.capitalize()}')
            
        plt.tight_layout()
        return fig

    def plot_customer_lifecycle(self):
        """Plot customer lifecycle visualizations"""
        # Create tenure segments
        self.data['tenure_segment'] = pd.qcut(self.data['tenure'], 
                                            q=4, 
                                            labels=['New', 'Early', 'Established', 'Loyal'])
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Customer Value Throughout Lifecycle
        sns.boxplot(data=self.data, x='tenure_segment', y='MonthlyCharges', 
                   hue='Churn', ax=axes[0])
        axes[0].set_title('Customer Value Throughout Lifecycle')
        
        # Average Services by Tenure
        services = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport']
        # Convert Yes/No to 1/0
        service_data = self.data[services].copy()
        for col in services:
            service_data[col] = (service_data[col] == 'Yes').astype(int)
        service_counts = service_data.groupby(self.data['tenure_segment'], observed=True).mean()
        service_counts.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Average Services by Tenure')
        
        # Contract Type Evolution
        contract_tenure = pd.crosstab(self.data['tenure_segment'], 
                                    self.data['Contract'], 
                                    normalize='index')
        contract_tenure.plot(kind='bar', stacked=True, ax=axes[2])
        axes[2].set_title('Contract Type Evolution')
        
        plt.tight_layout()
        return fig

    def plot_churn_analysis(self):
        """Plot churn analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Churn Rate by Contract Type
        sns.barplot(data=self.data, x='Contract', y='Churn', 
                   ax=axes[0, 0])
        axes[0, 0].set_title('Churn Rate by Contract Type')
        
        # Monthly Charges Distribution
        sns.boxplot(data=self.data, x='Churn', y='MonthlyCharges', 
                   ax=axes[0, 1])
        axes[0, 1].set_title('Monthly Charges Distribution by Churn')
        
        # Churn Rate by Internet Service
        sns.barplot(data=self.data, x='InternetService', y='Churn', 
                   ax=axes[1, 0])
        axes[1, 0].set_title('Churn Rate by Internet Service')
        
        # Tenure Distribution
        sns.boxplot(data=self.data, x='Churn', y='tenure', 
                   ax=axes[1, 1])
        axes[1, 1].set_title('Tenure Distribution by Churn')
        
        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self):
        """Plot correlation matrix for numeric features"""
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        correlation = numeric_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numeric Features')
        
        return plt.gcf()