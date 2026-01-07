import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from transformers import AutoTokenizer
from models.bert_classifier import BERTMultiLabelClassifier
import joblib


class SupportTicketPredictor:
    """
    Predictor class for customer support ticket classification
    """

    def __init__(self, model_path='models/best_model.pt', device = None):

        """
        Initislize the predictor
        
        Args:
            model_path (str): Path to trained model checkpoint
            device: torch device (None = auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load label encoder
        encoder_path = Path('data/processed/label_encoder.pkl')
        self.mlb = joblib.load(encoder_path)
        self.label_classes = list(self.mlb.classes_)
        self.num_classes = len(self.label_classes)

        print(f"Loaded {self.num_classes} classe: {self.label_classes}")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 128

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = BERTMultiLabelClassifier(num_classes = self.num_classes)
        checkpoint = torch.load(model_path, map_location = self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print("✅ Model loaded and ready for predictions!\n")


    def preprocess_text(self, text):
        """
        Preprocess and tokenize input text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Tokenized inputs
        """

        # Clean text
        text = text.lower().strip()

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation = True,
            padding = 'max_length',
            max_length = self.max_length,
            return_tensors = 'pt'
        )

        return encoding
    

    def predict(self, text, threshold = 0.5):
        """
        Predict categories for input text
        
        Args:
            text (str): Support ticket text
            threshold (float): Classification threshold
            
        Returns:
            dict: Prediction woth probabilities
        """

        # Preprocess
        encoding = self.preprocess_text(text)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)

        # Convert to numpy
        probs = probabilities.cpu().numpy()[0]
        predictions = (probs > threshold).astype(int)

        # get predicted categories
        predicted_labels = [self.label_classes[i] for i, pred in enumerate(predictions) if pred == 1]

        # Create results dictionary
        results = {
            'text': text,
            'predicted_categories': predicted_labels,
            'probabilities':{
                self.label_classes[i]: float(probs[i])
                for i in range(len(self.label_classes))
            },
            'threshold': threshold
        }

        return results

    
    def predict_batch(self, texts, threshold = 0.5):
        """
        Predict categories for multiple texs
        
        Args:
            texts (list): List of support ticket texts
            threshold (float): Classification threshold
            
        Returns:
            list: List of prediction dictionaries
        """

        results = []
        for text in texts:
            result = self.predict(text, threshold)
            results.append(result)
        return results

    def print_predictions(self, result):
        """
        Pretty print predictions results
        
        Aegs:
            result (dict): Prediction result from predict()
        """

        print(f"Text: {result['text']}")
        print(f"\n Predicted Categories: {', '.join(result['predicted_categories'])}")
        print(f"\n All Probabilities:")
        for category, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse = True):
            bar ='█' * int(prob *20)
            print(f" {category:20s}: {prob:.3f} {bar}")
        print()

        
def main():
    """ Main functions for testing predictions """

    # Initialize predictor
    predictor = SupportTicketPredictor()

    # Example tickets for testing
    test_tickets = [
        "I was charged twice for my subscription this month. Can you refund one charge?",
        "The app keeps crashing when I try to login. Error code 500.",
        "What are the shipping options for international orders?",
        "I can't access my account after changing my password.",
        "Do you have this product available in blue color?",
    ]

    print("="*70)
    print("TESTING PREDICTIONS ON SAMPLE TICKETS")
    print("="*70)

    # Make predictions
    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}:")
        print('='*70)

        result = predictor.predict(ticket, threshold = 0.5)
        predictor.print_predictions(result)

    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter support ticket text (or 'quit' to exit): \n")

    while True:
        user_input = input("Ticket:").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter some text. \n")
            continue

        result = predictor.predict(user_input)
        print()
        predictor.print_predictions(result)

if __name__ == "__main__":
    main()
