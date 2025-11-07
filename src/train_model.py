import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import time

from utils.data_loader import load_preprocessed_data
from data.dataset import create_data_loaders
from models.bert_classifier import create_model


class ModelTrainer:
    """
    Trainer class for BERT multi-label classification
    """

    def __init__(self, model, train_loader, val_loader, device, learning_rate = 2e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function for multi_label classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr = learning_rate)

        # Learning rate scheduler
        total_steps = len(train_loader)*10
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = total_steps
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """  Train for one epoch """
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc = "Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    

    def validate(self):
        """ Validate th model """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc = "validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()


        avg_loss = total_loss / len(self.val_loader)
        return avg_loss


    def train(self, num_epochs, save_path = 'model/best_model.pt'):
        """
        Full training loop

        Args: 
            num_epochs (int): Number of epochs to train
            save_path (str): Path to save the best model
        """

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")


        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")

            start_time = time.time()

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            epoch_time = time.time() - start_time


            print(f"\nEpoch Summary:")
            print(f" Train Loss: {train_loss:.4f}")
            print(f" Val Loss: {val_loss:.4f}")
            print(f" Time: {epoch_time:.2f}s")


            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                Path(save_path).parent.mkdir(parents = True, exist_ok = True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f" ✓ Saved best model (val_loss: {val_loss:.4f})")


        print(f"\n{'='*50}")
        print("Traing completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """ Main training function"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test, mlb = load_preprocessed_data()
    num_classes = len(mlb.classes_)

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        batch_size = 16,
        max_length = 128
    )

    # Create model
    print("\nCreating model...")
    model = create_model(num_classes = num_classes)

    # Create trainer
    trainer = ModelTrainer(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        device = device,
        learning_rate = 2e-5
    )

    # Train
    trainer.train(num_epochs = 3, save_path = 'models/best_model.pt')

    print("\n✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()