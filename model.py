from classes import ModelArchitecture, Loader
from var import *
from preprocessor import Normalizer
import os
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import re
from torch.utils.data import DataLoader
from nltk.stem import PorterStemmer
stemmer = PorterStemmer() 

ENTITY_RECOGNIZER_PATH = "models/active/EntityRecognizer.pth"
ORDER_RECOGNIZER_PATH = "models/active/OrderRecoginzer.pth"

LABELER_DATA_PATH = "database/labeler"
GROUPER_DATA_PATH = "database/grouper"

class PizzaSemanticParser:
    def __init__(self, parameters_folder, mode="labeler"):
        self.dataset_folder = LABELER_DATA_PATH if mode == "labeler" else GROUPER_DATA_PATH
        self.parameters_folder = parameters_folder
        self.load_model_parameters(labeler_mode=(True if mode == "labeler" else False))
        self.model = ModelArchitecture(self.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, self.output_size, self.word2idx)

    def load_data(self, train=True):
        with open(f"{self.dataset_folder}/x_{"train" if train else "dev"}.txt", 'r') as fx, open(f"{self.dataset_folder}/y_{"train" if train else "dev"}.txt", 'r') as fy:
            sentences = fx.read().strip().split('\n')
            labels = fy.read().strip().split('\n')
            fx.close()
            fy.close()
        return [sentence.split(",") for sentence in sentences], [label.split(',') for label in labels]

    def required_files_available(self):
        required_files = [f"{self.parameters_folder}word2idx.txt", f"{self.parameters_folder}label2idx.txt"]
        for file in required_files:
            if not os.path.isfile(file):
                return False
        return True

    def read_parameters_files(self):
        self.word2idx = {}
        self.label2idx = {}
        self.idx2label = {}

        with open(f"{self.parameters_folder}/word2idx.txt", "r") as file:
            for line in file:
                word, idx = line.strip().split(",")
                self.word2idx[word] = int(idx)
            file.close()

        with open(f"{self.parameters_folder}/label2idx.txt", "r") as file:
            for line in file:
                label, idx = line.strip().split(",")
                self.label2idx[label] = int(idx)
                self.idx2label[int(idx)] = label
            file.close()

        self.vocab_size = len(self.word2idx.keys())
        self.output_size = len(self.label2idx.keys())
    
    def extract_parameters(self, labeler_mode=True):
        def load_vocabulary():
            with open(f"{self.dataset_folder}/vocabulary.txt", 'r') as fv:
                # NOTE: every line consists of 2 parameters (the token, its frequency) so for now i igone the frequency
                vocab = [v.strip().split(",")[0] for v in fv if v!= ""]
                fv.close()
            return vocab

        print("Loading Data, Please Wait...")
        _, train_y  = self.load_data(train=True)
        print("Loading Vocabulary, Please Wait...")
        vocabulary = load_vocabulary()
        label_vocab = {label for label_list in train_y for label in label_list}

        self.word2idx = {word: idx + 2 for idx, word in enumerate(sorted(vocabulary))}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1

        self.vocab_size     = len(vocabulary)
        self.output_size    = len(label_vocab)

        self.label2idx      = {label: idx for idx, label in enumerate(sorted(label_vocab))}
        self.idx2label      = {idx: label for label, idx in self.label2idx.items()}

        os.makedirs(os.path.dirname(self.parameters_folder), exist_ok=True)

        print("Writing Model Parameters, Please Wait...")
        with open(f"{self.parameters_folder}/word2idx.txt", "w") as file:
            file.write("\n".join([f"{word},{idx}" for word, idx in self.word2idx.items()]))
            file.close()
        
        with open(f"{self.parameters_folder}/label2idx.txt", "w") as file:
            file.write("\n".join([f"{label},{idx}" for label, idx in self.label2idx.items()]))
            file.close()

    def load_model_parameters(self, labeler_mode=True):
        if self.required_files_available():
            print("Found required files... loading start")
            self.read_parameters_files()
        else: 
            print("Required files not found... extracting start")
            self.extract_parameters(labeler_mode)
        
        print("Model Parameters Were Loaded Successfully")

    def load_model_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def to_evaluation_mode(self):
        self.model.eval()

    def to_training_mode(self):
        self.model.train()

    def make_data_loader(self, x, y):
        dataset = Loader(x, y, self.word2idx, self.label2idx)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def train_model(self, epochs=10):
        train_sentences, train_labels = self.load_data(train=True)
        train_loader = self.make_data_loader(train_sentences, train_labels)

        self.to_training_mode()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0
            for sentences, labels in train_loader:
                sentences, labels = sentences, labels
                
                optimizer.zero_grad()
                predictions = self.model(sentences)

                loss = criterion(predictions.view(-1, self.output_size), labels.view(-1)) 
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / len(train_loader):.25f}")
        torch.save(self.model.state_dict(), f"output/model.pt")
        print("Training Done")
    
    def evaluate_model(self):
        dev_sentences, dev_labels = self.load_data(train=False)
        dev_loader = self.make_data_loader(dev_sentences, dev_labels)
        self.to_evaluation_mode()

        all_predictions = []
        all_labels = []

        total, correct = 0, 0

        with torch.no_grad():
            for sentences, labels in dev_loader:
                sentences, labels = sentences, labels
                predictions = self.model(sentences).argmax(dim=-1)
                
                # Accuracy calculation
                total += labels.numel()
                correct += (predictions == labels).sum().item()

                # Collect predictions and true labels for F1 score calculation
                all_predictions.extend(predictions.view(-1).tolist())
                all_labels.extend(labels.view(-1).tolist())

        # Calculate accuracy
        accuracy = correct / total

        # Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='weighted')  # Use 'weighted' for handling label imbalance

        # Print both metrics
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"F1 Score: {f1:.2f}")
    
    def preprocess_sentence(self, sentence):
        normalizer = Normalizer()
        preprocessed = normalizer.normalize(sentence)

        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word, tag in preprocessed]  
        indices = indices[:50] + [0] * (50 - len(indices)) 

        return torch.tensor([indices]), preprocessed

    def predict(self, sentence: str):
        input_tensor, preprocessed = self.preprocess_sentence(sentence)
        with torch.no_grad():
            output = self.model(input_tensor)  # Get logits
            predictions = output.argmax(dim=-1).squeeze(0)  # Get label indices
        return [self.idx2label[idx.item()] for idx in predictions if idx.item() in self.idx2label], preprocessed



# psp = PizzaSemanticParser("cpu", "database/models/labeler_model/", "labeler")   
# psp.load_model_weights("models/active/Entity.pth")
# psp.evaluate_model()