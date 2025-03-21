# Generative AI for Synthetic Data Generation

## Project Overview
This project demonstrates how to use Generative Adversarial Networks (GANs) and related models to generate synthetic tabular datasets. The project also implements privacy-preserving training techniques and tools for bias detection in generated data.

## Tech Stack
- PyTorch
- SDV (Synthetic Data Vault)
- Opacus (Differential Privacy)
- SDMetrics (Synthetic Data Evaluation)
- FairLearn (Fairness Evaluation)
- Hugging Face Transformers (for future enhancements)

## Installation
Install dependencies via:
```
pip install -r requirements.txt
```

## Usage
### 1. Generating Synthetic Data
```
python generate_data.py
```
This script will generate a synthetic dataset and save it as `synthetic_data.csv`.

### 2. Training Models with Privacy
Define your model and use `train_model_with_privacy` from `privacy_preserving.py` to train your model with differential privacy.
```
python privacy_preserving.py
```

### 3. Evaluating Bias
Run:
```
python bias_detection.py
```
This will compute synthesis quality and fairness metrics.

## Example Dataset
Create a CSV file named `real_data.csv` with a column named `label` for supervised learning.

## Future Work
- Improve model architectures.
- Enhance fairness evaluation metrics.
- Extend to other types of synthetic data generation.
- Integrate Hugging Face Transformers for enhanced model building.