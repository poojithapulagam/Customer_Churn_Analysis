from data_loader import DataLoader
from feature_processor import FeatureProcessor
from model_trainer import ModelTrainer
from visualizer import Visualizer
from business_analyzer import BusinessAnalyzer
from report_generator import ReportGenerator

def main():
    # Initialize data loader
    data_loader = DataLoader('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Load and prepare data
    print("Loading and preparing data...")
    data = data_loader.load_data()
    X, y = data_loader.prepare_features()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    
    # Get feature names
    feature_info = data_loader.get_feature_names()
    
    # Process features
    print("Processing features...")
    feature_processor = FeatureProcessor(
        categorical_features=feature_info['categorical'],
        numeric_features=feature_info['numeric']
    )
    
    X_train_processed = feature_processor.fit_transform(X_train)
    X_test_processed = feature_processor.transform(X_test)
    
    # Train and evaluate models
    print("Training models...")
    model_trainer = ModelTrainer()
    model_trainer.train_models(X_train_processed, y_train)
    
    print("Evaluating models...")
    evaluation_results = model_trainer.evaluate_models(X_test_processed, y_test)
    
    # Get feature importance
    feature_names = feature_processor.get_feature_names()
    feature_importance = model_trainer.get_feature_importance(feature_names)
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = Visualizer(data)
    
    # Plot and save visualizations
    viz_path = 'output/visualizations/'
    feature_importance_plot = visualizer.plot_feature_importance(feature_importance)
    feature_importance_plot.savefig(f'{viz_path}feature_importance.png')
    
    lifecycle_plot = visualizer.plot_customer_lifecycle()
    lifecycle_plot.savefig(f'{viz_path}customer_lifecycle.png')
    
    churn_analysis_plot = visualizer.plot_churn_analysis()
    churn_analysis_plot.savefig(f'{viz_path}churn_analysis.png')
    
    correlation_plot = visualizer.plot_correlation_matrix()
    correlation_plot.savefig(f'{viz_path}correlation_matrix.png')
    
    # Process all data for business analysis
    all_data_processed = feature_processor.transform(X)
    
    # Perform business analysis
    print("Performing business analysis...")
    best_model = model_trainer.trained_models['xgboost']  # Using XGBoost as our best model
    business_analyzer = BusinessAnalyzer(data, best_model, all_data_processed)
    business_metrics = business_analyzer.calculate_business_metrics()
    high_risk_segments = business_analyzer.identify_high_risk_segments()
    recommendations = business_analyzer.generate_recommendations()
    
    # Print results
    print("\nModel Evaluation Results:")
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name.capitalize()} Model:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])
    
    print("\nBusiness Metrics:")
    for metric, value in business_metrics.items():
        print(f"{metric}: {value:,.2f}")
        
    print("\nHigh Risk Segments:")
    for segment_type, probabilities in high_risk_segments.items():
        print(f"\n{segment_type.replace('_', ' ').title()}:")
        for category, prob in probabilities.items():
            print(f"{category}: {prob:.2%} churn probability")
    
    print("\nBusiness Recommendations:")
    for rec in recommendations:
        print(f"\nArea: {rec['area']}")
        print(f"Finding: {rec['finding']}")
        print(f"Recommendation: {rec['recommendation']}")
        print(f"Potential Impact: {rec['potential_impact']}")
        
    # Generate and save detailed report
    report = ReportGenerator.generate_report(
        evaluation_results, 
        business_metrics, 
        high_risk_segments, 
        recommendations
    )
    
    report_path = 'output/reports/churn_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()