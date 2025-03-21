import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def preprocess_data(dataframe):
    # Identify categorical columns
    categorical_columns = dataframe.select_dtypes(include=['object']).columns

    # Encode categorical columns
    label_encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        dataframe[column] = encoder.fit_transform(dataframe[column])
        label_encoders[column] = encoder

    # Separate features and labels
    X = dataframe.drop('label', axis=1).values
    y = dataframe['label'].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    return X_tensor, y_tensor


def train_model_with_privacy(real_data_path, epochs=10, batch_size=32):
    # Load Data
    data = pd.read_csv(real_data_path)
    X, y = preprocess_data(data)

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    # Define a simple neural network model
    model = nn.Sequential(
        nn.Linear(X.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    privacy_engine = PrivacyEngine()

    # Attach the privacy engine to the model
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    # Train the model
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = nn.BCELoss()(predictions, y_batch)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), 'privacy_preserving_model.pth')
    print("Privacy-preserving model trained and saved successfully!")

if __name__ == "__main__":
    train_model_with_privacy('synthetic_data.csv')
