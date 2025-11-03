import numpy as np
import joblib 
import os
from pathlib import Path


def load_preprocessed_data():
    """
    Load preprocessed train/validation/test splits and label encoder

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, mlb)
    """
    # Define paths
    data_dir = Path(__file__).parent.parent.parent / 'data'/ 'processed'

    required_files = [
        'X_train.npy', 'X_val.npy', 'X_test.npy',
        'y_train.npy', 'y_val.npy', 'y_test.npy',
        'label_encoder.pkl'
    ]

    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(f"Required file {file} not found in {data_dir}")
        
    # Load data splits, label, and label encoder 

    X_train = np.load(data_dir / 'X_train.npy', allow_pickle=True)
    X_val = np.load(data_dir / 'X_val.npy', allow_pickle=True)
    X_test = np.load(data_dir / 'X_test.npy', allow_pickle=True)

    y_train = np.load(data_dir / 'y_train.npy')    
    y_val = np.load(data_dir / 'y_val.npy')
    y_test = np.load(data_dir / 'y_test.npy')

    mlb = joblib.load(data_dir / 'label_encoder.pkl')

    print(f"Data loaded successfully:")
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Number of labels: {len(mlb.classes_)}")
    print(f"Label classes: {list(mlb.classes_)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, mlb


def get_data_info():
    """
    Get information about the loaded data without loading it

    Returns
        dict: Information about data shapes and classes
    """
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'

    # Load minimal info
    y_train = np.load(data_dir / 'y_train.npy')
    mlb = joblib.load(data_dir / 'label_encoder.pkl')

    return{
        'num_classes': len(mlb.clases_),
        'classes': list(mlb.classes_),
        'label_shape': y_train.shape[1]
    }
