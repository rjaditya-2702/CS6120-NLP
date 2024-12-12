import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification


class TorchDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        labels = self.dataframe.iloc[idx]['label'] 
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  
            padding='max_length',    # Pads the sequences to the max_length
            truncation=True,         # Truncates sequences longer than max_length
            return_tensors='pt',     # Returns PyTorch tensors
        )

        # Extract the tokenized input ids, attention mask
        input_ids = encoding['input_ids'].squeeze(0)  # Remove the batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Return the processed inputs and labels as tensors
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.float) 
        }


def predict_and_threshold(model, dataloader, device, threshold=0.5):
    """
    Function to predict and convert probabilities to labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get the inputs and labels
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass to get model outputs
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch_size, num_labels)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)  # Shape: (batch_size, num_labels)
            
            # Apply threshold to get binary predictions
            preds = (probs > threshold).int()  # Shape: (batch_size, num_labels)
            
            # Append predictions and true labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_preds, all_labels

# load dataset

# Example setup for model, tokenizer, and dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2ForSequenceClassification.from_pretrained("./code/model")

df = pd.read_csv('/Users/rjaditya/Documents/NEU-SEM/Fall-2024/NLP/project/CS6120-NLP/code/test.csv')
df = df[1000:]
test_dataset = TorchDataset(df, tokenizer) # convert into a dataset object for the dataloader to use.

# Set up DataLoader for the test set
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Get predictions and true labels
preds, true_labels = predict_and_threshold(model, test_dataloader, device)

# Flatten the arrays to compute the confusion matrix
true_labels_flat = true_labels.flatten()
preds_flat = preds.flatten()

# Generate the confusion matrix
cm = confusion_matrix(true_labels_flat, preds_flat)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(cm.shape[1]), yticklabels=np.arange(cm.shape[0]))

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Sentiment Classification')

# Save the plot as an image
plt.savefig('confusion_matrix.png')

# Optionally, display the plot
plt.show()