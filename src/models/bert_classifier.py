
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BERTMultiLabelClassifier(nn.Module):
    """
    BERT-based multi-label text classifier
    
    Args:
    model_name (str): Huggig Face BERT model name
    num_classes (int): Number of output classes
    dropout_rate (float): Dropout rate for regularization
    """

    def __init__(self, model_name = 'bert-base-uncased', num_classes = 7, dropout_rate = 0.3):
        super(BERTMultiLabelClassifier, self).__init__()

        # Load BERT configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

        # Store parameters
        self.num_classes = num_classes
        self.model_name = model_name
       
    def forward(self, input_ids, attention_mask):
        """
        Forward pass trough the model

        Args:
            input_ids (torch.Tensor): Tokenized inpit text
            attention_mask (torch.Tensor): Attention mask for padding

        Returns:
            torch.Tensor: Raw logits for each class
        """

        # Get BERT outputs
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        # Use [CLS] token representation
        pooled_output = outputs.pooler_output

        # Apply dropouts
        pooled_output = self.dropout(pooled_output)

        # Get logits
        logits = self.classifier(pooled_output)

        return logits
    

    def predict_proba(self, input_ids, attention_mask):
        """
        Get prediction probabilities using sigmoid activation

        Args: 
            input_ids (torch.Tensor): Tokenized input text
            attention_mask (torc.Tensor): Attention mask

        Returns:
            torch.Tensor: Probabilities for each class
        """


        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)

        return probabilities

    
    def predict(self, input_ids, attention_mask, threshold = 0.5):
        """
        Get binary predictions

        Args:
            input_ids (torch.Tensor): Tokenized input text
            attention_mask (torch.Tensor): Attention mask
            threshold (float): Classification threshold

        Returns:
            torch.Tensor: Binary predictions for each class
        """

        probabilities = self.predict_proba(input_ids, attention_mask)
        predictions = (probabilities > threshold).int()
        return predictions


def create_model(num_classes, model_name = 'bert-base-uncased', dropout_rate = 0.3):
    """
    Factory function to create the model

    Args:
        num_classes (int): Nmber of output classes
        model_name (str): Hugging Face model name
        dropout_rate (float): Dropout rate

    Returns:
        BERTMultiLabelClassifier: Initialized model
    """

    model = BERTMultiLabelClassifier(
        model_name = model_name,
        num_classes = num_classes,
        dropout_rate = dropout_rate
    )

    print(f"Model created:")
    print(f"- Base model: {model_name}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Dropout rate: {dropout_rate}")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model
    

if __name__ == "__main__":
    # Test model creation
    try:
        print("Testing model creation ...")
        model = create_model(num_classes = 7)

        # test forward pass with dummy data
        batch_size, seq_length = 2, 128
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        dummy_attention_mask = torch.ones(batch_size, seq_length)


        print("Testing forward pass ...")
        with torch.no_grad():
            outputs = model(dummy_input_ids, dummy_attention_mask)
            print(f"Output shape: {outputs.shape}")
            print(f"Expected shape: ({batch_size}, {model.num_classes})")

        print("✅ BERT classifier works correctly!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
