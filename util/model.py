import torch
import torch.nn as nn
from transformers import AutoModel

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

class Bert4Classify(nn.Module):
    def __init__(self, pretrained_model_name_or_path, dropout_rate, num_classes):
        super(Bert4Classify, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)
        d_model = 768 if 'bert' in pretrained_model_name_or_path else 1024
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, input_ids, att_mask):
        sentence_emb = self.get_sentence_embedding(input_ids, att_mask)
        output = self.classify(sentence_emb)
        return output
    
    def get_sentence_embedding(self, input_ids, att_mask):
        max_len = att_mask.sum(1).max()
        input_ids = input_ids[:, :max_len]
        att_mask = att_mask[:, :max_len]
        all_hidden = self.encoder(input_ids, att_mask)
        sentence_emb = all_hidden[0][:, 0]
        return sentence_emb

    def classify(self, x):
        output = self.mlp(x)
        return output
    
    def save_model(self, model_save_path):
        torch.save(self.state_dict(), model_save_path)
    
    def load_model(self, model_load_path):
        model_state_dict = torch.load(model_load_path)
        self.load_state_dict(model_state_dict)
        