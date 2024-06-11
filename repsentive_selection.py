from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_dataset
from tqdm import tqdm
import torch.optim as optim
import numpy as np
# confusion matrix
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
import random
from torch.nn import functional as F


label_desc = [
    "It's a world news",
    "It's a sports news",
    "It's a business news",
    "It's a science news"
]

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
sentence = 'Who are you voting for in 2020?'
# labels = ['business', 'art & culture', 'politics']
labels = ['This is a World','This is a sports news.', 'This is a business news.', 'This is a science news.']

# run inputs through model and mean-pool over the sequence
# dimension to get sequence-level representations
inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                     return_tensors='pt',
                                     pad_to_max_length=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
output = model(input_ids, attention_mask=attention_mask)[0]
sentence_rep = output[:1].mean(dim=1)
label_reps = output[1:].mean(dim=1)

# now find the labels with the highest cosine similarities to
# the sentence
similarities = F.cosine_similarity(sentence_rep, label_reps)
# soft_max 
similarities = F.softmax(similarities/0.1, dim=0)

print(similarities)