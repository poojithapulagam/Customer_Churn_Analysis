import pandas as pd

class BusinessAnalyzer:
    def __init__(self, data, model, X_processed):
        self.data = data
        self.model = model
        self.X_processed = X_processed
        self.predictions = model.predict(X_processed)
        self.prediction_probabilities = model.predict_proba(X_processed)
        
    def calculate_business_metrics(self):
        """Calculate key business metrics"""
        # Calculate current metrics
        total_customers = len(self.data)
        current_churn_rate = self.data['Churn'].map({'Yes': 1, 'No': 0}).mean()
        avg_monthly_revenue = self.data['MonthlyCharges'].mean()
        avg_customer_lifetime = self.data['tenure'].mean()
        
        # Calculate yearly impact
        yearly_lost_customers = total_customers * current_churn_rate
        yearly_lost_revenue = yearly_lost_customers * avg_monthly_revenue * 12
        
        # Calculate potential improvements
        target_churn_reduction = 0.2  # Target 20% reduction in churn
        improved_yearly_lost_customers = yearly_lost_customers * (1 - target_churn_reduction)
        improved_yearly_lost_revenue = improved_yearly_lost_customers * avg_monthly_revenue * 12
        potential_savings = yearly_lost_revenue - improved_yearly_lost_revenue
        
        return {
            'total_customers': total_customers,
            'current_churn_rate': current_churn_rate,
            'avg_monthly_revenue': avg_monthly_revenue,
            'avg_customer_lifetime': avg_customer_lifetime,
            'yearly_lost_customers': yearly_lost_customers,
            'yearly_lost_revenue': yearly_lost_revenue,
            'potential_savings': potential_savings
        }
        
    def identify_high_risk_segments(self):
        """Identify customer segments with high churn risk"""
        # Add prediction probabilities to data
        data_with_proba = self.data.copy()
        data_with_proba['churn_probability'] = self.prediction_probabilities[:, 1]
        
        # Analyze churn probability by different segments
        segments = {
            'contract_type': data_with_proba.groupby('Contract')['churn_probability'].mean(),
            'internet_service': data_with_proba.groupby('InternetService')['churn_probability'].mean(),
            'payment_method': data_with_proba.groupby('PaymentMethod')['churn_probability'].mean(),
            'monthly_charges': data_with_proba.groupby(pd.qcut(data_with_proba['MonthlyCharges'], 4))['churn_probability'].mean()
        }
        
        return segments
        
    def generate_recommendations(self):
        """Generate business recommendations based on analysis"""
        segments = self.identify_high_risk_segments()
        
        recommendations = []
        
        # Contract-based recommendations
        if segments['contract_type']['Month-to-month'] > 0.3:
            recommendations.append({
                'area': 'Contract Strategy',
                'finding': 'High churn rate in month-to-month contracts',
                'recommendation': 'Offer incentives for longer-term contracts',
                'potential_impact': 'Reduce churn rate by 20-30% in this segment'
            })
            
        # Service-based recommendations
        if 'Fiber optic' in segments['internet_service'] and segments['internet_service']['Fiber optic'] > 0.3:
            recommendations.append({
                'area': 'Service Quality',
                'finding': 'Higher churn in fiber optic service customers',
                'recommendation': 'Improve fiber optic service reliability and support',
                'potential_impact': 'Reduce technical service complaints by 40%'
            })
            
        # Payment-based recommendations
        if 'Electronic check' in segments['payment_method'] and segments['payment_method']['Electronic check'] > 0.3:
            recommendations.append({
                'area': 'Payment Methods',
                'finding': 'Higher churn rate with electronic check payments',
                'recommendation': 'Promote automatic payment methods',
                'potential_impact': 'Increase payment reliability by 25%'
            })
            
        return recommendations