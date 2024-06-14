# %load_ext cudf.pandas  # pandas operations now use the GPU!
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
# from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
from sklearn.utils import resample
from util.datasets import load_dataset

dataset = load_dataset("fancyzhx/ag_news")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define custom dataset class
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Prepare datasets
max_length = 128
df_train=pd.DataFrame(dataset['train'])
# # sample 5000 from each label
df_train_sampled = df_train.groupby('label').apply(lambda x: x.sample(5000)).reset_index(drop=True)
train_texts = df_train_sampled['text']
train_labels = df_train_sampled['label']
test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

train_dataset = AGNewsDataset(train_texts, train_labels, tokenizer, max_length)
test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# with out sample train 
train_texts = df_train['text']
train_labels = df_train['label']
train_dataset_all = AGNewsDataset(train_texts, train_labels, tokenizer, max_length)
train_loader_all = DataLoader(train_dataset_all, batch_size=32, shuffle=True)
# # Define LSTM model
# 
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_output, (hidden, cell) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return self.softmax(output)
    # save 
    def save(self, path):
        torch.save(self.state_dict(), path)
# # Hyperparameters
vocab_size = tokenizer.vocab_size
embed_size = 128
hidden_size = 256
output_size = 4  # Number of classes in AG News
num_layers = 2
dropout = 0.5

# Initialize model, loss function, and optimizer
model = LSTMModel(vocab_size, embed_size, hidden_size, output_size, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# # Training loop
# num_epochs = 3

# def train_model(model, device, train_loader, optimizer, criterion, epoch):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)
#         outputs = model(input_ids, attention_mask)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     avg_loss = total_loss / len(train_loader)
#     print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

# def evaluate_model(model, device, test_loader):
#     model.eval()
#     predictions, true_labels = [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)
#             outputs = model(input_ids, attention_mask)
#             _, preds = torch.max(outputs, dim=1)
#             predictions.extend(preds.cpu().numpy())
#             true_labels.extend(labels.cpu().numpy())
#     accuracy = accuracy_score(true_labels, predictions)
#     print(f'Test Accuracy: {accuracy:.4f}')
#     return accuracy

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Generate noisy labels using softmax outputs
# def get_softmax_out(model, loader, device):
#     model.eval()
#     softmax_out = []
#     with torch.no_grad():
#         for batch in loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             outputs = model(input_ids, attention_mask)
#             softmax_out.append(outputs.cpu().numpy())
#     return np.vstack(softmax_out)
# train_dataset = train_dataset_all
# softmax_out_avg = np.zeros([len(train_dataset), 4])
# softmax_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# for epoch in range(1, num_epochs + 1):
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     train_model(model, device, train_loader, optimizer, criterion, epoch)
#     softmax_out_avg += get_softmax_out(model, softmax_loader, device)

# softmax_out_avg /= num_epochs
# if not os.path.exists('ag_new_noise'):
#     os.makedirs('ag_new_noise')
# np.save('ag_new_noise/softmax_out_avg.npy', softmax_out_avg)
# model.save('ag_new_noise/model.pt')
print('Generating noisy labels according to softmax_out_avg...')

softmax_out_avg = np.load('ag_new_noise/softmax_out_avg.npy')
labels = np.array(train_dataset.labels)
label_noisy_cand, label_noisy_prob = [], []
for i in range(len(labels)):
    pred = softmax_out_avg[i,:].copy()
    pred[labels[i]] = -1
    label_noisy_cand.append(np.argmax(pred))
    label_noisy_prob.append(np.max(pred))

noise_rate = 0.4

label_noisy = labels.copy()
index = np.argsort(label_noisy_prob)[-int(noise_rate * len(labels)):]
label_noisy[index] = np.array(label_noisy_cand)[index]
text=train_dataset.texts
save_pth = os.path.join('ag_new_noise', 'dependent' + str(noise_rate) + '.csv')
pd.DataFrame.from_dict({'text':text,'label': labels, 'label_noisy': label_noisy
                        }).to_csv(save_pth, index=False)


# print('Noisy label data saved to', save_pth)

# noise_rate = 0.0
# label_noisy = labels.copy()
# index = np.argsort(label_noisy_prob)[-int(noise_rate * len(labels)):]
# label_noisy[index] = np.array(label_noisy_cand)[index]
# text=train_dataset.texts
# save_pth = os.path.join('AG_NEWS_NOISE', 'dependent' + str(noise_rate) + '.csv')
# pd.DataFrame.from_dict({'text':text,'label': labels, 'label_noisy': label_noisy
#                         }).to_csv(save_pth, index=False)
# print('Noisy label data saved to', save_pth)