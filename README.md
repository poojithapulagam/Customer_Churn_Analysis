# Customer Churn Analysis Project

A comprehensive machine learning project that analyzes customer churn patterns in a telecommunications company using the Telco Customer Churn dataset. This project provides end-to-end analysis from data loading to business recommendations.

## 🚀 Features

- **Complete ML Pipeline**: Data loading, preprocessing, feature engineering, and model training
- **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost
- **Professional Visualizations**: Feature importance, customer lifecycle, churn analysis, and correlation matrix
- **Business Intelligence**: ROI analysis, high-risk segment identification, and actionable recommendations
- **Comprehensive Reporting**: Detailed markdown reports with performance metrics and business insights

## 📊 Project Structure

```
Customer_Churn_Analysis/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset
├── src/
│   ├── main.py                                 # Main orchestrator
│   ├── data_loader.py                          # Data loading and preprocessing
│   ├── feature_processor.py                    # Feature engineering
│   ├── model_trainer.py                        # ML model training
│   ├── visualizer.py                           # Data visualization
│   ├── business_analyzer.py                    # Business insights
│   ├── report_generator.py                     # Report generation
│   └── advanced_analyzer.py                    # Advanced analytics
├── output/
│   ├── reports/
│   │   └── churn_analysis_report.md           # Comprehensive analysis report
│   └── visualizations/
│       ├── feature_importance.png              # Feature importance plot
│       ├── customer_lifecycle.png             # Customer lifecycle analysis
│       ├── churn_analysis.png                 # Churn pattern analysis
│       └── correlation_matrix.png             # Feature correlation matrix
├── requirements.txt                            # Python dependencies
├── customer_churn_analysis.ipynb              # Jupyter notebook
└── README.md                                  # This file
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/poojithapulagam/Customer_Churn_Analysis.git
   cd Customer_Churn_Analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Run the Complete Analysis
```bash
python src/main.py
```

### Run Jupyter Notebook
```bash
jupyter notebook customer_churn_analysis.ipynb
```

## 📈 Results

### Model Performance
- **Logistic Regression**: 78.75% accuracy, 83.19% ROC AUC
- **Random Forest**: 78.46% accuracy, 81.53% ROC AUC
- **XGBoost**: 77.40% accuracy, 81.08% ROC AUC

### Key Insights
- **Overall Churn Rate**: 26.58%
- **Annual Lost Revenue**: $1,453,294
- **Potential Savings**: $290,659 (with 20% churn reduction)
- **High-Risk Segments**:
  - Month-to-month contracts: 42.8% churn probability
  - Fiber optic users: 41.95% churn probability
  - Electronic check payments: 45.14% churn probability

### Business Recommendations
1. **Contract Strategy**: Offer incentives for longer-term contracts
2. **Service Quality**: Improve fiber optic service reliability
3. **Payment Methods**: Promote automatic payment methods

## 📊 Output Files

After running the analysis, the following files are generated:

- `output/reports/churn_analysis_report.md` - Comprehensive analysis report
- `output/visualizations/feature_importance.png` - Feature importance visualization
- `output/visualizations/customer_lifecycle.png` - Customer lifecycle analysis
- `output/visualizations/churn_analysis.png` - Churn pattern analysis
- `output/visualizations/correlation_matrix.png` - Feature correlation matrix

## 🛠️ Technologies Used

- **Python** 3.8+
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Jupyter Notebook** for interactive analysis

## 📝 Dataset

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle, which contains information about telecom customers and their churn status.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project demonstrates a complete machine learning pipeline for customer churn analysis, from data preprocessing to business recommendations. The analysis provides actionable insights that can help telecom companies reduce customer churn and improve business performance.
