import pandas as pd
from sdmetrics.single_table import NewRowSynthesis
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.preprocessing import LabelEncoder

def preprocess_data(dataframe):
    """Encodes categorical columns for comparison."""
    label_encoders = {}
    for column in dataframe.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        dataframe[column] = encoder.fit_transform(dataframe[column])
        label_encoders[column] = encoder
    return dataframe

def evaluate_bias(real_data_path: str, synthetic_data_path: str):
    # Load real and synthetic data
    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    
    # Preprocess both datasets
    real_data = preprocess_data(real_data)
    synthetic_data = preprocess_data(synthetic_data)
    
    # Ensure both datasets have the same columns
    if list(real_data.columns) != list(synthetic_data.columns):
        print("Columns mismatch between real and synthetic data. Please check your files.")
        return

    # Evaluate Synthesis Quality Metric
    synthesis_metric = NewRowSynthesis.compute(real_data, synthetic_data)
    print(f"\nSynthesis Quality Metric (Similarity Score): {synthesis_metric}\n")

    # Check if 'label' and 'gender' columns are present for bias detection
    if 'label' in real_data.columns and 'label' in synthetic_data.columns and 'gender' in real_data.columns:
        y_true = real_data['label']
        y_pred = synthetic_data['label']
        sensitive_features = real_data['gender']
        
        # Calculate Demographic Parity Difference
        fairness_metric = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        
        # Print the fairness metric
        print(f"Fairness Metric (Demographic Parity Difference): {fairness_metric}\n")
    else:
        print("Label or sensitive feature column not found for bias detection.")

if __name__ == "__main__":
    evaluate_bias('real_data.csv', 'synthetic_data.csv')
