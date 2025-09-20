class ReportGenerator:
    @staticmethod
    def generate_report(evaluation_results, business_metrics, high_risk_segments, recommendations):
        report = """
# Customer Churn Analysis Report

## 1. Model Performance Analysis

### Model Evaluation Results:
"""
        for model_name, results in evaluation_results.items():
            report += f"""
#### {model_name.capitalize()} Model:
- Accuracy: {results['accuracy']:.4f}
- ROC AUC: {results['roc_auc']:.4f}

Classification Report:
```
{results['classification_report']}
```
"""

        report += """
## 2. Business Metrics Analysis

### Key Performance Indicators:
"""
        for metric, value in business_metrics.items():
            if 'rate' in metric:
                report += f"- {metric.replace('_', ' ').title()}: {value:.2%}\n"
            else:
                report += f"- {metric.replace('_', ' ').title()}: ${value:,.2f}\n"

        report += """
## 3. High Risk Segment Analysis

### Churn Probability by Segment:
"""
        for segment_type, probabilities in high_risk_segments.items():
            report += f"\n#### {segment_type.replace('_', ' ').title()}:\n"
            for category, prob in probabilities.items():
                report += f"- {category}: {prob:.2%} churn probability\n"

        report += """
## 4. Business Recommendations

### Action Items:
"""
        for rec in recommendations:
            report += f"""
#### {rec['area']}:
- Finding: {rec['finding']}
- Recommendation: {rec['recommendation']}
- Potential Impact: {rec['potential_impact']}
"""

        report += """
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
"""
        return report