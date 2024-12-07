import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Loader(Dataset):
    def __init__(self, sentences, labels, word2idx, label2idx, max_len=50):
        self.label2idx = label2idx
        self.sentences = [[word2idx.get(word, word2idx['<UNK>']) for word in sentence] for sentence in sentences]
        self.labels = [[label2idx[label] for label in label_list] for label_list in labels]
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        sentence = sentence[:self.max_len] + [0] * (self.max_len - len(sentence))
        label = label[:self.max_len] + [self.label2idx['NONE']] * (self.max_len - len(label))
        return torch.tensor(sentence), torch.tensor(label)


class ModelArchitecture(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, word2idx):
        super(ModelArchitecture, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx['<PAD>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                            num_layers=2, 
                            bidirectional=True, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        predicted = self.fc(lstm_out)
        return predicted
    
