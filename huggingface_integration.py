# huggingface_integration.py

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

AVAILABLE_MODELS = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased'
}

def preprocess_data(dataframe):
    label_encoders = {}
    for column in dataframe.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        dataframe[column] = encoder.fit_transform(dataframe[column])
        label_encoders[column] = encoder

    # Convert dataframe to text representation
    text_data = dataframe.apply(lambda row: ' '.join(map(str, row.values)), axis=1).tolist()
    labels = dataframe['label'].values.tolist()
    
    return text_data, labels


def load_data(real_data_path, synthetic_data_path):
    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    real_texts, real_labels = preprocess_data(real_data)
    synthetic_texts, synthetic_labels = preprocess_data(synthetic_data)

    return (real_texts, real_labels), (synthetic_texts, synthetic_labels)


def evaluate_model(model, tokenizer, texts, labels):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs).logits
        predictions = torch.sigmoid(outputs).squeeze().detach().cpu().numpy()
        predictions = (predictions > 0.5).astype(int)
        accuracy = (predictions == labels).mean()
    return accuracy


def train_and_evaluate_model(real_data_path, synthetic_data_path, model_name):
    (real_texts, real_labels), (synthetic_texts, synthetic_labels) = load_data(real_data_path, synthetic_data_path)

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(AVAILABLE_MODELS[model_name], num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(AVAILABLE_MODELS[model_name])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(3):
        inputs = tokenizer(real_texts, padding=True, truncation=True, return_tensors='pt')
        labels = torch.tensor(real_labels).unsqueeze(1).float()
        outputs = model(**inputs).logits
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/3, Loss: {loss.item()}")

    real_accuracy = evaluate_model(model, tokenizer, real_texts, real_labels)
    synthetic_accuracy = evaluate_model(model, tokenizer, synthetic_texts, synthetic_labels)

    print(f"{model_name.upper()} - Real Data Accuracy: {real_accuracy}")
    print(f"{model_name.upper()} - Synthetic Data Accuracy: {synthetic_accuracy}")

    return {
        'Model': model_name,
        'Real Accuracy': real_accuracy,
        'Synthetic Accuracy': synthetic_accuracy
    }


if __name__ == "__main__":
    results = {}
    for model_name in AVAILABLE_MODELS.keys():
        results[model_name] = train_and_evaluate_model('real_data.csv', 'synthetic_data.csv', model_name)
