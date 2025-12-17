import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    hamming_loss,
    classification_report,
    multilabel_confusion_matrix
)

import pandas as pd
from tqdm import tqdm

from utils.data_loader import load_preprocessed_data
from data.dataset import create_data_loaders
from models.bert_classifier import BERTMultiLabelClassifier


class ModelEvaluator:
    """
    Evaluator for multi-label classification model
    """

    def __init__(self, model, test_loader, device, label_classes):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.label_classes = label_classes
        self.model.eval()

    def predict(self, threshold = 0.5):
        """
        Get predictions on test set
        
        Args: 
            threshold (float): Classification threshold
            
        Returns:
            tuple: (predictions, true_labels, probabilities)
        """

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Get Logits
                logits = self.model(input_ids, attention_mask)

                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(logits)

                # Apply thresgold for binary predictions
                predictions = (probabilities > threshold).int()

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())

        # Concatenate all batches
        predictions = np.vstack(all_predictions)
        true_labels = np.vstack(all_labels)
        probabilities = np.vstack(all_probabilities)

        return predictions, true_labels, probabilities
        

    def calculate_metrics(self, predictions, true_labels):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            predictions: Binary predictions
            true_labels: True labels
            
        Returns:
            dict: Dictionary of metrics
        """

        # Overall metrics
        hamming = hamming_loss(true_labels, predictions)

        # per-sample accuracy (exact match)
        exact_match = accuracy_score(true_labels, predictions)

        # Per-label metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predictions,
            average = 'weighted',
            zero_division=0
        )

        # Mocro and macro averages
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average='micro',
            zero_division=0
        )

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average='macro',
            zero_division=0
        )

        metrics = {
            'hamming_loss': hamming,
            'exact_match_ratio': exact_match,
            'precision_weighted':precision,
            'recall_weighted':recall,
            'f1_weighted':f1,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
        }

        return metrics
    
    def per_class_metrics(self, predictions, true_labels):
        """
        Calculate metrics for each class
        
        Args:
            predictions: Binary predictions
            true_labels: True labels
            
        Returns:
            pd.DataFrame: Per-class metrics
        """

        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predictions,
            average=None,
            zero_division=0
        )

        results = []
        for i, class_name in enumerate(self.label_classes):
            results.append({
                'Class':class_name,
                'Precision': precision[i],
                'Recall':recall[i],
                'F1-score':f1[i],
                'Support':support[i]
            })

        df = pd.DataFrame(results)
        return df

    
    def print_evaluation_report(self, metrics, per_class_df):
        """
        Print comprehensive evaluation report
        """

        print("\n" + "="*70)
        print("MODEL EVALUATION REPORT")
        print("="*70)

        print("\n Overall Metrics:" )
        print(f" Exact match Ratio: {metrics['exact_match_ratio']:.4f}")
        print(f" Hamming loss: {metrics['hamming_loss']:.4f}")

        print("\n Micro Average (Overall performence):")
        print(f" Precisionn: {metrics['precision_micro']:.4f}")
        print(f" Recall: {metrics['recall_micro']:.4f}")
        print(f" F1-Score: {metrics['f1_micro']:.4f}")

        print("\n Macro Averages (Per-Class Average):")
        print(f"Precision: {metrics['precision_macro']:.4f}")
        print(f"Recall: {metrics['recall_macro']:.4f}")
        print(f"F1_Score: {metrics['f1_macro']:.4f}")

        print("\n Weighted Averages (Support-Weighted):")
        print(f" Precision: {metrics['precision_weighted']:.4f}")
        print(f" Recall: {metrics['recall_weighted']:.4f}")
        print(f" F1-Score: {metrics['f1_weighted']:.4f}")

        print("\n Per-Class Performance:")
        print(per_class_df.to_string(index=False))

        print("\n" + "="*70)

    
def load_trained_model(model_path, num_classes, device):
    """
    Load trained model from checkpoint

    Args:
        model_path (str): Path to model chackpoint
        num_classes (int): Number of classes
        device: torch device

    Returns:
        model: Loaded model
    """

    print(f"Loading model from {model_path}...")

    # Create model
    model = BERTMultiLabelClassifier(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict']) 

    print(f"✅ Model Loaded successfully")
    print(f" Trained for {checkpoint['epoch']+1} epochs")
    print(f" Best validation loss: {checkpoint['val_loss']:.4f}")

    return model


def main():
    """ Main evaluation function"""

    #Configuration
    MODEL_PATH = 'models/best_model.pt'
    THRESHOLD = 0.5

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load preprocessed data
    print("\n Loading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test, mlb = load_preprocessed_data()
    num_classes = len(mlb.classes_)
    label_classes = list(mlb.classes_)

    # Create test data loader
    print("\nCreating test data loader...")
    _,_, test_loader, tokenizer = create_data_loaders(
        X_train[:10], X_val[:10], X_test,
        y_train[:10], y_val[:10], y_test,
        batch_size = 16, 
        max_length = 128
    )

    # Load trained model
    model = load_trained_model(MODEL_PATH, num_classes, device)

    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, device, label_classes)

    # Get predictions
    print("\n Generating predictions on test set...")
    predictions, true_labels, probabilities = evaluator.predict(threshold=THRESHOLD)

    # Calculate metrics
    print("\n Calculating metrics...")
    metrics = evaluator.calculate_metrics(predictions, true_labels)
    per_class_metrics = evaluator.per_class_metrics(predictions, true_labels)

    # Print report
    evaluator.print_evaluation_report(metrics, per_class_metrics)

    # Save results
    print("\n Saving evaluation results...")
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Save per_class metrics
    per_class_metrics.to_csv('results/per_class_metrics.csv', index = False)

    # Save overall metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('results/overall_metrics.csv', index = False)

    # Save predictions
    np.save('results/test_predicions.npy', predictions)
    np.save('results/test_probabilities.npy', probabilities)
    np.save('results/test_true_labels.npy', true_labels)

    print("✅ Evaluation complete! Results saved to 'results/' directory")    

if __name__ =="__main__":
    main()
  