import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer

def generate_synthetic_data(real_data_path: str, model_type='CTGAN'):
    # Load real data
    real_data = pd.read_csv(real_data_path)
    
    # Define metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)

    # Initialize model based on user selection
    if model_type == 'CTGAN':
        model = CTGANSynthesizer(metadata)
    elif model_type == 'TVAE':
        model = TVAESynthesizer(metadata)
    elif model_type == 'GaussianCopula':
        model = GaussianCopulaSynthesizer(metadata)
    else:
        raise ValueError("Unsupported model type. Choose 'CTGAN', 'TVAE', or 'GaussianCopula'.")
    
    # Fit the model with the real data
    model.fit(real_data)
    
    # Generate synthetic data
    synthetic_data = model.sample(len(real_data))
    
    return synthetic_data

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data('real_data.csv', model_type='CTGAN')
    synthetic_data.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data generated and saved successfully!")
