from libraries import *
from test_creator import TESTGoldOutputGenerator
from preprocessor import Normalizer

class JsonUtils:
    @staticmethod
    def log(obj):
        # Pretty JSON strings
        json_str = json.dumps(obj, indent=4)

        # Create Panels for each JSON
        panel = Panel(json_str, title="Order", expand=True)

        # Render them side by side
        console = Console()
        console.print(Columns([panel]))

    @staticmethod
    def compare(obj1, obj2):
        # Pretty JSON strings
        json1_str = json.dumps(obj1, indent=4)
        json2_str = json.dumps(obj2, indent=4)

        # Create Panels for each JSON
        panel1 = Panel(json1_str, title="EXPECTED", expand=True)
        panel2 = Panel(json2_str, title="GOT", expand=True)

        # Render them side by side
        console = Console()
        console.print(Columns([panel1, panel2]))

    @staticmethod
    def is_equal(obj1, obj2):
            if isinstance(obj1, dict) and isinstance(obj2, dict):
                if set(obj1.keys()) != set(obj2.keys()):
                    return False
                return all(JsonUtils.is_equal(obj1[key], obj2[key]) for key in obj1)
            elif isinstance(obj1, list) and isinstance(obj2, list):
                if len(obj1) != len(obj2):
                    return False

                unmatched = obj2[:]
                for item in obj1:
                    for candidate in unmatched:
                        if JsonUtils.is_equal(item, candidate):
                            unmatched.remove(candidate)
                            break
                    else:
                        return False
                return True
            else:
                return obj1 == obj2


def is_num(x):
    if Normalizer().replace_numbers(x).endswith("NUM"):
        return True
    else:
        return False

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


class RNNSequenceLabeling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, word2idx):
        super(RNNSequenceLabeling, self).__init__()
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
    

class TestsetLoader:
    idx : int
    sentences: list[str]
    gold_outputs: list[dict]


    def read_file(self):
        file = open("./database/PIZZA_test.json", "r")

        for line in file:
            obj = json.loads(line.strip())
            self.sentences.append(obj["test.SRC"])
            self.converter.preprocess(obj["test.TOP"])
            self.gold_outputs.append(self.converter.y)
        
        file.close()
    
    def count(self):
        return len(self.sentences)

    def empty(self):
        return self.idx >= len(self.sentences) 

    def fetch_testcase(self):
        self.idx += 1
        return self.sentences[self.idx - 1], self.gold_outputs[self.idx - 1]

    def __init__(self):
        self.idx = 0
        self.converter = TESTGoldOutputGenerator()
        self.sentences = []
        self.gold_outputs = []
        self.read_file()