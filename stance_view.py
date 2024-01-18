#!/usr/bin/env python
# coding: utf-8

# In[25]:


# get_ipython().system('pip install nltk')
# get_ipython().system('pip install torchtext')


# In[26]:


import pandas as pd
import numpy as np
import os

import nltk
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import torchtext
from torchtext.data import get_tokenizer 
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset
from torch.nn import LSTM,Embedding,Dropout,GRU,RNN
from torch import sigmoid, cat

import matplotlib.pyplot as plt
print("Success!")


# In[27]:


DATASET_PATH = "./"
train_bodies = pd.read_csv(os.path.join(DATASET_PATH, 'data_combined.csv'))
train_bodies.head()


# In[28]:


data = train_bodies.copy()
data['stance_cat'] = data['Stance'].map({'agree':0,'disagree':1,'discuss':2,'unrelated':3}).astype(int)
data['stance_cat'].value_counts()


# In[29]:


corpus = np.r_[data['Headline'].values,data['articleBody'].values]
print(49972*2)
print(len(corpus)) # first 49972 contains the Headline and next 49972 contains the articleBody

vocabulary = []
for sentence in corpus:
    vocabulary.extend(sentence.split(' '))
    # print(sentence)

vocabulary = list(set(vocabulary))
vocab_length = len(vocabulary)
print("Vocabulary Length is {0}".format(vocab_length))


# In[30]:


max_features = 5000
MAX_NB_WORDS = 24000
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 128


# In[31]:


from torchtext.vocab import GloVe

glove = GloVe(name="6B", dim=50)

def setup_embedding_index():
    embedding_index = {}
    for word, idx in glove.stoi.items():
        embedding_vector = glove.vectors[idx]
        embedding_index[word] = embedding_vector.numpy()
    return embedding_index

embeddings_index = setup_embedding_index()
# print(embeddings_index.shape)


# In[32]:



# Convert the 'Headline' column in the 'data' DataFrame to a list of strings
headlines = data['Headline'].str.lower().tolist()

# Use the 'get_tokenizer' function from torchtext to tokenize the headlines
tokenizer = get_tokenizer('basic_english')
headline_tokens = [tokenizer(headline) for headline in headlines]

# Build a vocabulary from the tokenized text data
vocab_headline = build_vocab_from_iterator(headline_tokens, min_freq=3, specials=['<pad>', '<unk>'])
vocab_headline.set_default_index(vocab_headline['<unk>'])

# Get the length of the headline vocabulary (total unique words + 2 for special tokens)
vocab_headline_length = len(vocab_headline)
print(vocab_headline_length)
word_index_headline = [torch.tensor([vocab_headline[token] for token in tokens]) for tokens in headline_tokens]
# print(111)
NUM_WORDS_HEADLINE = vocab_headline_length
# print(222)
# print(NUM_WORDS_HEADLINE)
# Pad headlines to length 16
padded_headlines = pad_sequence([headline[:16] for headline in word_index_headline], batch_first=True, padding_value=0)
print(padded_headlines)


# In[33]:


body_s = data['articleBody'].str.lower().tolist()

body_tokens = [tokenizer(body) for body in body_s]

# Build a vocabulary from the tokenized text data
vocab_body = build_vocab_from_iterator(body_tokens, min_freq=3, specials=['<pad>', '<unk>'])
vocab_body.set_default_index(vocab_body['<unk>'])
# print(vocab_body)
# Get the length of the headline vocabulary (total unique words + 2 for special tokens)
vocab_body_length = len(vocab_body)
print(vocab_body_length)
word_index_body = [torch.tensor([vocab_body[token] for token in tokens]) for tokens in body_tokens]

NUM_WORDS_HEADLINE = vocab_body_length
print(NUM_WORDS_HEADLINE)

padded_body = pad_sequence([body[:48] for body in word_index_body], batch_first=True, padding_value=0)
print(padded_body)


# In[34]:


embedding_matrix_headline = torch.zeros((vocab_headline_length, EMBEDDING_DIM))
# print(embedding_matrix_headline)
for word, i in vocab_headline.vocab.get_stoi().items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_headline[i] = torch.tensor(embedding_vector)
dims = embedding_matrix_headline.shape[1]
print(embedding_matrix_headline)
print(dims)


# In[35]:


embedding_matrix_body = torch.zeros((vocab_body_length, EMBEDDING_DIM))

for word, i in vocab_body.vocab.get_stoi().items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_body[i] = torch.tensor(embedding_vector)

dims = embedding_matrix_body.shape[1]

print(dims)


# In[36]:


# print(headline_tensor.shape)
print(padded_headlines.shape)
print(padded_body.shape)
# print(body_tensor.shape)
headline_tensor = padded_headlines
body_tensor = padded_body
print(body_tensor)


# In[37]:


import torch
from sklearn.model_selection import train_test_split

# Assuming you have tensors for padded_docs_headline, padded_docs_body, and labels

# Convert the numpy arrays to PyTorch tensors
# padded_docs_headline = torch.tensor(headline_tensor)
# padded_docs_body = torch.tensor(padded_docs_body)
labels = torch.tensor(data["stance_cat"])

# Split the data and labels into training and testing sets
padded_docs_headline_train, padded_docs_headline_test, padded_docs_body_train, padded_docs_body_test, labels_train, labels_test = train_test_split(headline_tensor, 
                                             body_tensor, 
                                             labels, 
                                             test_size=0.1, 
                                             random_state=42)


# In[38]:


import torch
import torch.nn as nn

class LSTM_Text(nn.Module):
    def __init__(self, vocab_headline_length, vocab_body_length, embedding_matrix_headline, embedding_matrix_body, embedding_dim=EMBEDDING_DIM, lstm_units=64, dropout_rate=0.25, num_classes=4):
        super(LSTM_Text, self).__init__()

        # Embedding layers
        self.embedding_headline = nn.Embedding(vocab_headline_length, embedding_dim)
        self.embedding_body = nn.Embedding(vocab_body_length, embedding_dim)

        # Initialize the embedding layers with pre-trained word embeddings
        self.embedding_headline.from_pretrained(torch.FloatTensor(embedding_matrix_headline))
        self.embedding_body.from_pretrained(torch.FloatTensor(embedding_matrix_body))

        # LSTM layer for headline and body
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Output layer
        # self.output_layer = nn.Linear(lstm_units, num_classes)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim*64, 100),
            nn.Linear(100, num_classes),
        )


    def forward(self, headline, body):
        # Embedding the input sequences for headline and body
        embedded_headline = self.embedding_headline(headline)
        embedded_body = self.embedding_body(body)

        # Concatenate the embedded headline and body
        output = torch.cat((embedded_headline, embedded_body), dim=1)

        # LSTM layer
        output, _ = self.lstm(output)
        output = torch.flatten(output, 1)
        
        # Apply dropout
        # lstm_output = self.dropout(lstm_output)

        # Output layer
        output = self.output_layer(output)


        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# In[44]:



from torch.utils.data import TensorDataset, DataLoader
# inputs_headline = torch.LongTensor(padded_docs_headline_train)
# inputs_body = torch.LongTensor(padded_docs_body_train)
# targets = torch.LongTensor(labels_train)
# Create an instance of the LSTM_Text model
# embedding_matrix_headline = torch.LongTensor(embedding_matrix_headline)
# embedding_matrix_body = torch.LongTensor(embedding_matrix_body)
num_epochs = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

model = LSTM_Text(vocab_headline_length, vocab_body_length, embedding_matrix_headline, embedding_matrix_body)
model = model.to(device)

# device = 'mps' if torch.has_mps() else 'cpu' 
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.009, momentum=0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0)

# Define the directory to save the model checkpoints
MODELS_DIR = "./model/"

# Define a variable to keep track of the best validation accuracy
best_val_accuracy = 0.91
# Convert training and validation data to PyTorch tensors
train_data = TensorDataset(padded_docs_headline_train, padded_docs_body_train, labels_train)
valid_data = TensorDataset(padded_docs_headline_test, padded_docs_body_test, labels_test)

# Define batch size (you can adjust this based on your memory capacity)
# batch_size = 128
batch_size = 32

# Create DataLoader objects for training and validation data
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size)

# map_location=torch.device('cpu')
model.load_state_dict(torch.load("./model/49-0.89.pth",map_location = torch.device('cpu')))

# Training loop
for epoch in range(num_epochs):
    print("\n epoch:" , epoch)
    avg_train_loss = 0
    # Training code here...
    for inputs_headline, inputs_body, targets in train_dataloader:
        inputs_headline = inputs_headline.to(device)
        inputs_body = inputs_body.to(device)
        targets = targets.to(device)
        # print("inputs_headline.shape",inputs_headline.shape)
        # print("inputs_body.shape",inputs_body.shape)
        # Forward pass
        output = model(inputs_headline, inputs_body)
        # Calculate the loss
        loss = criterion(output, targets)
        # print(loss)
        avg_train_loss += loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"average train loss:{avg_train_loss/len(train_dataloader)} train loss:{loss:.4f}")
    
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        total_true_positives = [0] * 4
        total_predicted_positives = [0] * 4  
        total_actual_positives = [0] * 4
        for inputs_headline, inputs_body, targets in valid_dataloader:
            inputs_headline = inputs_headline.to(device)
            inputs_body = inputs_body.to(device)
            targets = targets.to(device)
            
            # Forward pass
            output = model(inputs_headline, inputs_body)
            output = torch.softmax(output, dim=1)
            # Calculate validation accuracy
            _, predicted = torch.max(output, 1)
            # predicted = output
            total_correct += (predicted == targets).sum().item() 
            total_samples += targets.size(0)
            
            # Calculate true positives and actual positives for each class
            for cls in range(4):
                true_positives = ((predicted == cls) & (targets == cls)).sum().item()
                predicted_positives = (predicted == cls).sum().item()
                actual_positives = (targets == cls).sum().item()
                
                total_true_positives[cls] += true_positives
                total_predicted_positives[cls] += predicted_positives
                total_actual_positives[cls] += actual_positives

        val_accuracy = total_correct / total_samples
        print("accuary: ",val_accuracy)
        # Calculate recall for each class
        recall_per_class = [tp / (ap + 1e-6) for tp, ap in zip(total_true_positives, total_actual_positives)]
        print("recall per class:", recall_per_class)

        # Calculate precision for each class
        precision_per_class = [tp / (pp + 1e-6) for tp, pp in zip(total_true_positives, total_predicted_positives)]
        print("precision per class:", precision_per_class)

        # Calculate F1-score for each class
        f1_score_per_class = [2 * p * r / (p + r + 1e-6) for p, r in zip(precision_per_class, recall_per_class)]
        print("f1-score per class:", f1_score_per_class)

# Check if the current validation accuracy is better than the best so far
# if val_accuracy > best_val_accuracy:
#     # Save the model checkpoint to the specified filepath
#     filename = os.path.join(MODELS_DIR, f"{epoch:02d}-{val_accuracy:.2f}.pth")
#     torch.save(model.state_dict(), filename)

    # Update the best validation accuracy
    # best_val_accuracy = val_accuracy
    # print("best_val_accuracy: ",best_val_accuracy)
       


# In[ ]:




