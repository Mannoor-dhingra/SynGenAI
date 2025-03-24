# 🔥 Generative AI for Synthetic Data Generation

## 🚀 Overview
Hey there! 👋 This project is all about using Generative AI to create synthetic tabular datasets, training privacy-preserving models, and checking for biases in the generated data. We’re even throwing in some Hugging Face Transformers (BERT, RoBERTa, DistilBERT) to compare how well models trained on synthetic data stack up against real data. Sounds fun, right? Let’s dive in!

## 📂 Project Structure
```
📁 Your Repo
├── generate_data.py            # Generates synthetic tabular data
├── privacy_preserving.py       # Trains models with differential privacy
├── bias_detection.py           # Checks for biases between real and synthetic data
├── huggingface_integration.py  # Fine-tunes and evaluates Hugging Face models
├── requirements.txt            # Dependencies
├── real_data.csv               # The real dataset
├── synthetic_data.csv          # The generated synthetic dataset
├── README.md                   # You are here!
```

## 🔧 Installation
First things first, get your environment ready:
```bash
pip install -r requirements.txt
```
You’ll need packages like `SDV`, `Opacus`, `FairLearn`, and `transformers`. Make sure they install properly.

## 📌 How to Use
This repo is split into multiple steps, and you can use whichever ones you like:

### 1. Generate Synthetic Data
```bash
python generate_data.py
```
Creates a file called `synthetic_data.csv` that mimics your real data.

### 2. Train Privacy-Preserving Models
```bash
python privacy_preserving.py
```
Trains a neural network using differential privacy. You’ll get a saved model that’s privacy-compliant.

### 3. Check for Bias & Fairness
```bash
python bias_detection.py
```
Measures how similar the synthetic data is to the real data and checks for biases using demographic parity difference.

### 4. Hugging Face Model Training
```bash
python huggingface_integration.py
```
Evaluates `BERT`, `RoBERTa`, and `DistilBERT` on both real and synthetic data. Want to see how well your synthetic data holds up against the real deal? This is your go-to script.

## 📊 Results So Far
| Model      | Real Data Accuracy | Synthetic Data Accuracy |
|------------|--------------------|-------------------------|
| BERT       | 0.47               | 0.522                   |
| RoBERTa    | 0.47               | 0.522                   |
| DistilBERT | 0.53               | 0.478                   |

DistilBERT seems to handle the synthetic data a bit better, but overall, there’s definitely room for improvement. The cool part? The synthetic data is holding its own against the real stuff!

## 💡 What’s Next
- Try different architectures and training methods.
- Enhance bias detection with new metrics.
- Compare models trained with and without differential privacy.
- Automate everything with a pipeline.

## 📜 License
MIT License. Use this project, improve it, break it, remix it – just keep it open! 😊

---

