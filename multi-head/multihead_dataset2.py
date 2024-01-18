#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
data = pd.read_csv('./dataset/full_data.csv')
# d = data[:15000]
# d.head


import re
def parse_feature_string_(s):
    """Parse the feature string and return a list of floats."""
    s = s.strip("[]")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r",+", ",", s)
    return [float(x) for x in s.split(",")]

# d = data[5:7]
parsed_features = data['emotion_feature'].apply(parse_feature_string_)

# Convert the parsed features (list of lists) to a PyTorch tensor
tensor_emotion = torch.tensor(parsed_features.tolist(), dtype=torch.float64)



# In[3]:


import ast

def parse_feature_string(feature_str):
    # Replace newline characters with spaces
    feature_str = feature_str.replace('\n', ' ')
    # print(feature_str)
    # Split the string by spaces, filter out empty strings, and then join with commas
    modified_str = ','.join(filter(None, feature_str.split(' ')))
    # print(modified_str)
    # Use ast.literal_eval to convert the modified string to a list
    float_list = ast.literal_eval('[' + modified_str + ']')
    
#    Convert scientific notation strings to float
    float_values = [val for val in float_list]
    # print(float_values)
    # Round to 8 decimal places
    # rounded_values = [round(val, 8) for val in float_values]
    
    return float_values


# In[4]:


d = data[5:7]
parsed_features = data['style_feature'].apply(parse_feature_string)

# Convert the parsed features (list of lists) to a PyTorch tensor
tensor_style = torch.tensor(parsed_features.tolist(), dtype=torch.float64).squeeze(1)

parsed_features,
parsed_features, tensor_style




# # d = data[5:7]
parsed_features = data['stance_feature'].apply(parse_feature_string)

# Convert the parsed features (list of lists) to a PyTorch tensor
tensor_stance = torch.tensor(parsed_features.tolist(), dtype=torch.float64).squeeze(1)

# # parsed_features,
# parsed_features, tensor_stance




# In[8]:


#concat the style and stance feature tensor
# feature_tensor = torch.cat((tensor_style, tensor_stance), dim=1)
feature_tensor = tensor_stance
print(feature_tensor.shape)
print(feature_tensor.dtype)


# In[24]:


# Convert feature_tensor to float32 dtype
feature_tensor = feature_tensor.to(dtype=torch.float64)

feature_tensor.dtype


# In[17]:


feature_tensor[0][1].item()


# In[9]:



# Convert to PyTorch tensors
X = feature_tensor
y = torch.tensor(data['label'].values, dtype=torch.float64)

# Split into training and validation sets
train_size = int(0.7 * len(X))
val_size = len(X) - train_size

X_train, X_val = torch.split(X, [train_size, val_size])
y_train, y_val = torch.split(y, [train_size, val_size])

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=128)

# X.shape, y.shape
# X.dtype, y.dtype


# In[10]:


import torch.nn as nn
import torch.optim as optim

# Hyperparameters
N_EMB = 64
HIDDEN_DIM = 64
OUTPUT_DIM = 1
DROPOUT = 0.7

# Define the modified model
class ModifiedModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(ModifiedModel, self).__init__()
        
        self.fc_input = nn.Linear(input_dim, N_EMB)
        self.bn_input = nn.BatchNorm1d(N_EMB)
        self.lstm = nn.LSTM(N_EMB, hidden_dim, bidirectional=True, batch_first=True)
        self.mha = nn.MultiheadAttention(2*hidden_dim, num_heads=8)
        self.fc1 = nn.Linear(2*hidden_dim, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(256, 32)
        # self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(128, output_dim)
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.fc_input(text)
        embedded = self.bn_input(embedded)
        # print(embedded.shape)
        lstm_out, _ = self.lstm(embedded.unsqueeze(1))
        # print(lstm_out.shape)
        attn_output, _ = self.mha(lstm_out, lstm_out, lstm_out)
        # print(attn_output)
        # print(attn_output.shape)
        x = attn_output.squeeze(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = F.relu(self.bn3(x))

        return self.fc4(x)




# In[11]:


import torch.nn.functional as F

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # Round predictions to the closest integer (0 or 1)
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # Convert into float for division
    acc = correct.sum() / len(correct)
    return acc



# In[13]:

# according the different features change the input_dim
# model = ModifiedModel(input_dim=4, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
# model = ModifiedModel(input_dim=6, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
# model = ModifiedModel(input_dim=32, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
model = ModifiedModel(input_dim=42, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.to(torch.float64)
#不可以1e-3
LEARNING_RATE = 1e-4
# Define the loss function and the optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary classification
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Training loop
# num_epochs = 100


# In[14]:


import os
# Re-run the training loop with accuracy computation
training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []
y_true = []
y_pred = []
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_acc = 0
    for text, labels in train_loader:
        optimizer.zero_grad()
        # text, labels = batch
        text, labels = text.to(device), labels.to(device)
        # predictions = model(text).squeeze(1)
        # print(next(model.parameters()).dtype)
        # print(text.dtype)

        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels.float())
        acc = binary_accuracy(predictions, labels.float())
        

         # optimizer.step()
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
    training_losses.append(total_loss/len(train_loader))
    training_accuracies.append(total_acc/len(train_loader))


    print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}, Training Accuracy: {total_acc/len(train_loader)}")

   # Validation loop
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for text, labels in val_loader:
            # text, labels = batch
            text = text.to(device)
            labels = labels.to(device)
            predictions = model(text).squeeze(1)

            # print(predictions.shape)
            loss = criterion(predictions, labels.float())
            acc = binary_accuracy(predictions, labels.float())
            val_loss += loss.item()
            val_acc += acc.item()
            y_true.append(labels.cpu().numpy())
            y_pred.append((predictions > 0.5).float().cpu().numpy())
            
    test_labels = np.concatenate(y_true, axis=0)  # Concatenate all true labels in training
    test_predictions = np.concatenate(y_pred, axis=0)  # Concatenate all predictions in training
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions)
    test_recall = recall_score(test_labels, test_predictions)
    test_f1 = f1_score(test_labels, test_predictions)
    
    print(f"Testing Metrics - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")

    print(f"       Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_acc/len(val_loader)}")


if( (val_acc/len(val_loader)) > 0.8) :
       # Save the model checkpoint to the specified filepath
            filename = os.path.join('./model', f"{epoch:02d}-val{(val_acc/len(val_loader)):.2f}.pth")
            torch.save(model.state_dict(), filename)
# In[ ]:


def countParameters(model):
    """ Counts the total number of trainiable parameters in the model """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# In[ ]:


trainable, frozen = countParameters(model)
print(model)
_LOGGER('wwwwwwwwwwwwww', 'mode infomation:', model_shape=model.shape, tb=trainable)
print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")


# In[ ]:
counter = 0
def _LOGGER(tag, info, **kwargs):
    global counter
    counter += 1
    print('===============================================================')
    print(f'EISSegNet Counter: {counter}')
    print(f'TAG: {tag}')
    print(f'INFO: {info}')
    for key, value in kwargs.items():
        print(f"kwarg: {key}: {value}")
    print('===============================================================')



