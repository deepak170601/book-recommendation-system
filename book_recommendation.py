# Cell 1: Data Collection
import pandas as pd
import numpy as np
from paraphraser import Paraphraser

# Assuming you have a CSV file named 'books.csv' with columns 'title', 'description'
df = pd.read_csv('books.csv')

# Display the first few rows of the dataset
df.head()

# Cell 2: Data Preprocessing with Transformers and Paraphraser
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

# Initialize the paraphraser
paraphraser = Paraphraser()

# Handle missing values in the 'description' column
df['description'].fillna('', inplace=True)

# Paraphrase descriptions to be within 512 tokens
df['paraphrased_description'] = [paraphraser.paraphrase(description, max_length=512) for description in df['description']]

# Tokenize paraphrased book descriptions using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sequences = [tokenizer.encode(description, add_special_tokens=True) for description in df['paraphrased_description']]

# Pad sequences to a fixed length
max_sequence_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length, dtype='long', truncating='post', padding='post')

# Display the tokenized sequences
print(X[:5])

# Cell 3: Model Architecture with Transformers
from transformers import BertModel, BertConfig
import torch
import torch.nn as nn

# Define a custom model using BERT as an embedding layer
class BERTAutoencoder(nn.Module):
    def __init__(self):
        super(BERTAutoencoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, max_sequence_length)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state)

# Build the model
model = BERTAutoencoder()

# Display the model summary
print(model)

# Cell 4: Model Training
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X)

# Create DataLoader
data_loader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 5
for epoch in range(epochs):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        attention_mask = (inputs != 0)  # Create attention mask to ignore padded tokens
        outputs = model(inputs, attention_mask=attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Cell 5: Recommendation
# Take user input from console
user_query = input("Enter a book description: ")

# Tokenize and encode user input using BERT tokenizer
user_sequence = tokenizer.encode(user_query, add_special_tokens=True)
user_padded = pad_sequences([user_sequence], maxlen=max_sequence_length, dtype='long', truncating='post', padding='post')
user_input_tensor = torch.tensor(user_padded)

# Get reconstructed sequence
attention_mask_user = (user_input_tensor != 0)
reconstructed_sequence = model(user_input_tensor, attention_mask=attention_mask_user)

# Store reconstructed sequence in the original DataFrame
df['reconstructed_sequence'] = [model(torch.tensor(sequence), attention_mask=(torch.tensor(sequence) != 0)).detach().numpy() for sequence in X]

# Calculate similarity and recommend books
df['similarity'] = df['reconstructed_sequence'].apply(
    lambda x: np.linalg.norm(np.array(x) - np.array(reconstructed_sequence.detach().numpy()))
)

# Sort by similarity to user input
recommended_books = df.sort_values(by='similarity').head(12)[['title', 'similarity']]

print("Recommended Books:")
print(recommended_books)
