
# Customer Churn Analysis Report

## 1. Model Performance Analysis

### Model Evaluation Results:

#### Logistic Model:
- Accuracy: 0.7875
- ROC AUC: 0.8319

Classification Report:
```
              precision    recall  f1-score   support

           0       0.83      0.89      0.86      1033
           1       0.62      0.52      0.56       374

    accuracy                           0.79      1407
   macro avg       0.73      0.70      0.71      1407
weighted avg       0.78      0.79      0.78      1407

```

#### Random_forest Model:
- Accuracy: 0.7846
- ROC AUC: 0.8153

Classification Report:
```
              precision    recall  f1-score   support

           0       0.82      0.90      0.86      1033
           1       0.63      0.47      0.54       374

    accuracy                           0.78      1407
   macro avg       0.73      0.69      0.70      1407
weighted avg       0.77      0.78      0.77      1407

```

#### Xgboost Model:
- Accuracy: 0.7740
- ROC AUC: 0.8108

Classification Report:
```
              precision    recall  f1-score   support

           0       0.83      0.87      0.85      1033
           1       0.59      0.50      0.54       374

    accuracy                           0.77      1407
   macro avg       0.71      0.69      0.69      1407
weighted avg       0.76      0.77      0.77      1407

```

## 2. Business Metrics Analysis

### Key Performance Indicators:
- Total Customers: $7,032.00
- Current Churn Rate: 26.58%
- Avg Monthly Revenue: $64.80
- Avg Customer Lifetime: $32.42
- Yearly Lost Customers: $1,869.00
- Yearly Lost Revenue: $1,453,294.21
- Potential Savings: $290,658.84

## 3. High Risk Segment Analysis

### Churn Probability by Segment:

#### Contract Type:
- Month-to-month: 42.80% churn probability
- One year: 10.43% churn probability
- Two year: 3.01% churn probability

#### Internet Service:
- DSL: 18.58% churn probability
- Fiber optic: 41.95% churn probability
- No: 7.59% churn probability

#### Payment Method:
- Bank transfer (automatic): 16.50% churn probability
- Credit card (automatic): 15.14% churn probability
- Electronic check: 45.14% churn probability
- Mailed check: 19.37% churn probability

#### Monthly Charges:
- (18.249, 35.588]: 11.30% churn probability
- (35.588, 70.35]: 24.24% churn probability
- (70.35, 89.862]: 37.04% churn probability
- (89.862, 118.75]: 33.41% churn probability

## 4. Business Recommendations

### Action Items:

#### Contract Strategy:
- Finding: High churn rate in month-to-month contracts
- Recommendation: Offer incentives for longer-term contracts
- Potential Impact: Reduce churn rate by 20-30% in this segment

#### Service Quality:
- Finding: Higher churn in fiber optic service customers
- Recommendation: Improve fiber optic service reliability and support
- Potential Impact: Reduce technical service complaints by 40%

#### Payment Methods:
- Finding: Higher churn rate with electronic check payments
- Recommendation: Promote automatic payment methods
- Potential Impact: Increase payment reliability by 25%

## 5. Visualization References

The following visualizations are available in the output/visualizations folder:
1. feature_importance.png - Shows the most influential features for churn prediction
2. customer_lifecycle.png - Displays customer behavior patterns throughout their lifecycle
3. churn_analysis.png - Presents detailed churn patterns across different segments
4. correlation_matrix.png - Shows relationships between different features

## 6. Next Steps

1. Implement recommended actions, prioritizing:
   - Contract strategy modifications
   - Service quality improvements
   - Payment method optimization

2. Monitor KPIs:
   - Monthly churn rate
   - Customer lifetime value
   - Service adoption rates
   - Customer satisfaction scores

3. Regular Review:
   - Monthly performance tracking
   - Quarterly strategy adjustment
   - Annual comprehensive analysis
