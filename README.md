# 🎫 Customer Support Ticket Classifier

A multi-label text classification system that automatically categorizes customer support tickets using a fine-tuned BERT model. This end-to-end machine learning project demonstrates the complete ML pipeline from data generation to deployment.

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/AnnaMm/customer-support-classifier)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

##  Live Demo

Try the interactive demo: [**Customer Support Classifier on HuggingFace Spaces**](https://huggingface.co/spaces/AnnaMm/customer-support-classifier)

##  Overview

This project implements a BERT-based neural network to classify customer support tickets into multiple categories simultaneously. The model can identify tickets that span multiple issues (e.g., a billing problem that also requires a refund).

### Categories

The system classifies tickets into 7 categories:

-  **Billing Issue** - Payment and billing problems
-  **Technical Problem** - App crashes, errors, bugs
-  **Account Access** - Login and authentication issues
-  **Product Inquiry** - Product questions and information
-  **Refund Request** - Refund and reimbursement requests
-  **Shipping Concern** - Delivery and shipping issues
-  **Service Complaint** - General service complaints

##  Features

- **Multi-label Classification**: Assigns multiple relevant categories to a single ticket
- **Real-time Predictions**: Fast inference with BERT-based model
- **Interactive Web Interface**: User-friendly Streamlit application
- **Adjustable Threshold**: Control prediction sensitivity
- **Probability Scores**: View confidence levels for each category

##  Model Performance

| Metric | Score |
|--------|-------|
| Exact Match Ratio |0.505 |
| Hamming Loss | 0.0921 |
| F1-Score (Micro) | 0.7318 |
| F1-Score (Macro) | 0.5602 |
| Precision (Weighted) | 0.8752 |
| Recall (Weighted) | 0.6153 |


##  Tech Stack

**Machine Learning:**
- PyTorch 2.0+
- Transformers (Hugging Face)
- Scikit-learn
- BERT (bert-base-uncased)

**Development:**
- Python 3.9+
- Streamlit
- NumPy, Pandas
- Jupyter Notebooks

**Deployment:**
- HuggingFace Spaces
- Git LFS

##  Project Structure

```
customer-support-classifier/
├── data/
│   ├── raw/                    # Raw ticket data
│   └── processed/              # Preprocessed data & label encoder
├── src/
│   ├── models/
│   │   └── bert_classifier.py  # BERT model architecture
│   ├── utils/
│   │   └── data_loader.py      # Data loading utilities
│   ├── data/
│   │   └── dataset.py          # PyTorch dataset classes
│   ├── predict.py              # Inference script
│   ├── train_model.py          # Training pipeline
│   └── evaluate_model.py       # Evaluation script
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_preprocessing.ipynb # Data preprocessing
├── app/
│   └── streamlit_app.py       # Streamlit web application
├── models/
│   └── best_model.pt          # Trained model checkpoint
├── scripts/
│   └── data_generation.py     # Data generation script
└── requirements.txt           # Python dependencies
```

##  Getting Started

### Prerequisites

- Python 3.9 or higher
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/A-nnaM/customer-support-classification.git
cd customer-support-classification
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the trained model**
- The model is hosted on HuggingFace due to size constraints
- Option A: Download from [HuggingFace Space Files](https://huggingface.co/spaces/AnnaMm/customer-support-classifier/tree/main/models)
- Option B: Use the live demo instead of running locally
- Place downloaded `best_model.pt` in the `models/` directory

### Usage

#### Run the Web Application (Local)

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

#### Make Predictions via Python

```python
from src.predict import SupportTicketPredictor

# Initialize predictor
predictor = SupportTicketPredictor()

# Make prediction
ticket = "I was charged twice for my subscription this month"
result = predictor.predict(ticket, threshold=0.3)

print(f"Predicted categories: {result['predicted_categories']}")
print(f"Probabilities: {result['probabilities']}")
```

#### Evaluate the Model

```bash
python src/evaluate_model.py
```

##  Model Training

The model was trained on 1000 realistic customer support tickets generated using ChatGPT.

**Training Configuration:**
- Base Model: BERT (bert-base-uncased)
- Epochs: 3
- Batch Size: 16
- Learning Rate: 2e-5
- Optimizer: AdamW
- Loss Function: BCEWithLogitsLoss
- Hardware: Google Colab (T4 GPU)
- Training Time: ~15 minutes

**To retrain the model:**

```bash
# 1. Generate/prepare data
python scripts/data_generation.py

# 2. Run preprocessing (or use notebooks)
jupyter notebook notebooks/02_preprocessing.ipynb

# 3. Train model (use Google Colab for GPU)
python src/train_model.py
```

##  How It Works

1. **Text Preprocessing**: Input text is cleaned and tokenized using BERT tokenizer
2. **Feature Extraction**: BERT encoder generates contextual embeddings
3. **Classification**: Fully connected layer outputs probability for each category
4. **Threshold Application**: Categories above threshold are selected
5. **Multi-label Output**: Returns all relevant categories with confidence scores

##  Future Improvements

- [ ] Expand dataset with more diverse examples
- [ ] Add FastAPI REST API
- [ ] Add more categories
- [ ] Improve model performance with hyperparameter tuning
- [ ] Active learning pipeline
- [ ] A/B testing framework
- [ ] Create more comprehensive documentation

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Author

**Anna M**
- GitHub: [@A-nnaM](https://github.com/A-nnaM)
- HuggingFace: [@AnnaMm](https://huggingface.co/AnnaMm)

##  Acknowledgments

- BERT model from [Hugging Face Transformers](https://huggingface.co/transformers/)
- Training infrastructure: [Google Colab](https://colab.research.google.com/)
- Deployment platform: [HuggingFace Spaces](https://huggingface.co/spaces)

##  Contact

For questions or feedback, please open an issue or reach out via GitHub.

---

⭐ If you find this project useful, please consider giving it a star!