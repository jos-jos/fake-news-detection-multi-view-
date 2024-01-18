# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('./dataset/fake_real_data.csv')
# d = data[:200]
# d.to_csv('./demo.csv')

# %%
import ast

def parse_feature_string_v(feature_str):
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

import re
def parse_feature_string(s):
    """Parse the feature string and return a list of floats."""
    s = s.strip("[]")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r",+", ",", s)
    return [float(x) for x in s.split(",")]

import re
# deal with the list of style feature haven't the ','
def convert_string_to_list(s):
    """Convert the string representation of the list to a proper Python list."""
    # Use regex to find all floats in the string
    floats = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    # Convert the strings to floats and return as a list
    return [float(x) for x in floats]
data['style_features'] = data['style_features'].apply(convert_string_to_list)


# %%
inconsistent_rows = data[data['style_features'].apply(len) != 8]
print(inconsistent_rows)


# %%
# Remove rows where the length of the list in 'style_features' is not 8
data = data[data['style_features'].apply(len) == 8]
print(data.shape)
# Now, convert the 'style_features' column to a tensor
tensor_style = torch.tensor(data['style_features'].tolist(), dtype=torch.float64)
print(tensor_style.shape)

# %%

data['stance_feature'] = data['stance_feature'].apply(parse_feature_string_v)


# %%
# inconsistent_rows = data[data['stance_feature'].apply(len) != 8]
# print(inconsistent_rows)


# %%
# Remove rows where the length of the list in 'style_features' is not 8
# data = data[data['style_features'].apply(len) == 8]
# print(data.shape)
# Now, convert the 'style_features' column to a tensor
tensor_stance= torch.tensor(data['stance_feature'].tolist(), dtype=torch.float64).squeeze(1)
print(tensor_stance.shape)


# %%
# d = data[5:7]
# parsed_features = data['style_feature'].apply(parse_feature_string)

# Convert the parsed features (list of lists) to a PyTorch tensor
# tensor_stance = torch.tensor(parsed_features.tolist(), dtype=torch.float64).squeeze(1)


# tensor_stance = torch.tensor(data['style_features'].tolist(), dtype=torch.float64)

# # parsed_features,
# parsed_features, tensor_stance


# %%
# tensor_stance[0][1].item()


# %%
# print(tensor_stance.shape)
# print(parsed_features.shape)

# print(data['emotion_feature'].head(10))

# %%
# d = data[5:7]
parsed_features = data['emotion_feature'].apply(parse_feature_string)
# # non_numeric_entries = [item for sublist in parsed_features.tolist() for item in sublist if not isinstance(item, (int, float))]
# # print(non_numeric_entries)

tensor_emotion = torch.tensor(parsed_features.tolist(), dtype=torch.float64)

# tensor_emotion.shape

# parsed_features, tensor_emotion


# %%
# one_hot_predictions = pd.get_dummies(data['predictions'])

# Convert the one-hot encoded DataFrame to a tensor
# tensor_shance = torch.tensor(one_hot_predictions.values, dtype=torch.float64)
# tensor_shance.shape


# %%
#concat the style and stance feature tensor
feature_tensor = torch.cat((tensor_stance, tensor_emotion,tensor_style), dim=1)
# print(feature_tensor.shape)
# print(feature_tensor.dtype)


# %%
# Convert feature_tensor to float32 dtype
feature_tensor = feature_tensor.to(dtype=torch.float64)

# feature_tensor.dtype


# %%
# feature_tensor[0][1].item()


# %%
# Convert to PyTorch tensors
X = feature_tensor
y = torch.tensor(data['label'].values, dtype=torch.float64)
print(X.shape)
# Split into training and validation sets
train_size = int(0.7 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = torch.split(X, [train_size, val_size,test_size])
y_train, y_val, y_test = torch.split(y, [train_size, val_size, test_size])

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
print(len(train_dataset))


train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=128)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128)

X.shape, y.shape
X.dtype, y.dtype


# %%
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
N_EMB = 64
HIDDEN_DIM = 64
OUTPUT_DIM = 1
DROPOUT = 0.7

# Define the Multi-head model
class MultiheadModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MultiheadModel, self).__init__()
        
        self.fc_input = nn.Linear(input_dim, N_EMB)
        self.lstm = nn.LSTM(N_EMB, hidden_dim, bidirectional=True, batch_first=True)
        self.mha = nn.MultiheadAttention(2*hidden_dim, num_heads=8)
        self.fc1 = nn.Linear(2*hidden_dim, 256)
        # self.fc2 = nn.Linear(256, output_dim)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_dim)
        # self.fc4 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.fc_input(text)
        # print(embedded.shape)
        lstm_out, _ = self.lstm(embedded.unsqueeze(1))
        # print(lstm_out.shape)
        attn_output, _ = self.mha(lstm_out, lstm_out, lstm_out)
        # print(attn_output)
        # print(attn_output.shape)
        x = attn_output.squeeze(1)
        # print(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(f"fc1:{x.shape}")
        # print(x.shape)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # print(f"fc2:{x.shape}")
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        # x = F.relu(x)

        return x





# %%
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


# %%
i = 0
for text, labels in train_loader:
    print(f"{i}text: {text}")
    print(f"{i}labels: {labels}")
    i+=1


# %%
# model = MultiheadModel(input_dim=4, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
# model = MultiheadModel(input_dim=6, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
# model = MultiheadModel(input_dim=12, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
# model = MultiheadModel(input_dim=18, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
# model = MultiheadModel(input_dim=10, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
model = MultiheadModel(input_dim=22, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.to(torch.float64)

# Define the loss function and the optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary classification
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Training loop
num_epochs = 100


# %%
import os
# Re-run the training loop with accuracy computation
y_true = []
y_pred = []
training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []
num_epochs = 100
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

        
            y_true.append(labels.cpu().numpy())
            y_pred.append((predictions > 0.2).float().cpu().numpy())
            # print(predictions.shape)
            loss = criterion(predictions, labels.float())
            acc = binary_accuracy(predictions, labels.float())
            val_loss += loss.item()
            val_acc += acc.item()
        
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
    filename = os.path.join('./model', f"{epoch:02d}-{val_acc:.2f}.pth")
    torch.save(model.state_dict(), filename)


# %%
def countParameters(model):
    """ Counts the total number of trainiable parameters in the model """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# %%
trainable, frozen = countParameters(model)
print(model)
print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")


# %%



