import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class SupportTicketDataset(Dataset):
    """
    PyTorch Dataset class for support ticket text classification
    
    Args:
        texts (array-like): Array of text strings
        labels (array-like): Binary encoded labels for multi-label classification
        tokenizer: Hugging Face tokenizer
        max_length (int): Maximum sequence length for tokenization
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing tokenized text and labels
        """
        # Get text and labels for this index
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Return dictionary with all necessary data
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, 
                       model_name='bert-base-uncased', max_length=128, 
                       batch_size=16, num_workers=0):
    """
    Create PyTorch DataLoaders for training, validation, and testing
    
    Args:
        X_train, X_val, X_test: Text data arrays
        y_train, y_val, y_test: Label arrays
        model_name (str): Hugging Face model name for tokenizer
        max_length (int): Maximum sequence length
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = SupportTicketDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = SupportTicketDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = SupportTicketDataset(X_test, y_test, tokenizer, max_length)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"DataLoaders created:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    # Test the dataset creation
    from src.utils.data_loader import load_preprocessed_data

    try:
        print("Loading preprocessed data...")
        X_train, X_val, X_test, y_train, y_val, y_test, mlb = load_preprocessed_data()

        print("Creating data loaders with small subset...")
        #Create data loaders (small batch for testing)
        train_loader, val_loader, test_loader, tokenizer = create_data_loaders(
            X_train[:10], X_val[:5], X_test[:5], 
            y_train[:10], y_val[:5], y_test[:5],
            batch_size = 2
        )

        print("Testing batch creation.. ")
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")

        print("Dataset and Dataloaders work correctly!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
