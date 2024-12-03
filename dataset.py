from var    import *
import re
import json

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

class Normalizer:
    ''' 
        Removes All Punctuation From The Given String 
        Punctuation will is defined in var.py
    '''
    def remove_punctuations(self, TOP):
        regex = "".join([ fr"\{punc}" for punc in PUNCTUATIONS ])
        TOP = re.sub(fr"[{regex}]", '', TOP)
        return TOP
    
    '''
        Removes Words From The Given String Which Are Defined In var.py
        Note: This Words Is Assumed To Be Stemmed Already
    '''
    def remove_words(self, TOP):
        regex = fr"(?<=(?:\b|^))(?:{"|".join(BLACKLIST)})(?=(?:\b|&))"
        return re.sub(fr"({regex})", '', TOP)

    def replace_numbers(self, TOP):
        regex = r'\b\d+\b'
        TOP = re.sub(regex, 'DIGIT', TOP)
        return TOP

    '''
        Follow Rules Of Replacing Some Words With Other In Case Of Needing
    '''
    def reconstruct_words(self, TOP):
        return TOP
    
    '''
        - Replace Multiple Spaces With Only One Space
        - Removes Spaces At The Beginning And End Of The Given String
    '''
    def reorganize_spaces(self, TOP):
        TOP = re.sub(r'\s+', ' ', TOP)
        return re.sub(r'(?:\s+$)|(?:^\s+)', '', TOP)
    
    '''
        Stems the given word and convert it to lowercase
    '''
    def stem_word(self, token):
        return stemmer.stem(token, to_lowercase=False)

    '''
        Stems the given sentence and convert all words to lowercase
    '''
    def stem_sentence(self, sentence):
        return " ".join([ self.stem_word(word) for word in sentence.split(' ') ])

    '''
        Normalizes the given sentence by performing the following steps:
        1. Reconstructs the words using the rules defined in var
        2. Removes punctuation defined in var
        3. Reorganizes spaces
        4. Stems the sentence
        5. Returns the normalized sentence
    '''
    def normalize(self, sentence):
        NEXT = self.remove_punctuations(sentence)
        NEXT = self.remove_words(NEXT)
        NEXT = self.replace_numbers(NEXT)
        NEXT = self.reconstruct_words(NEXT)
        NEXT = self.reorganize_spaces(NEXT)
        NEXT = self.stem_sentence(NEXT)
        return NEXT
    

class DatasetUtils:
    x: list[str]
    y: list[str]
    vocabulary: set
    # this is used to retrieve the leaf nodes with their entity name
    mapper              : dict
    # all groups of the entity (NOT)
    # ex: (NOT TOPPING BBQ) (NOT TOPPING sauce)
    not_groups          : dict
    # all groups of the entity (COMPLEX_TOPPING)
    complex_groups      : dict
    # all groups of the entity (PIZZAORDER)
    pizza_orders        : dict
    # all groups of the entity (DRINKORDER)
    drink_orders        : dict
    
    def __init__(self):
        self.normalizer = Normalizer()
        self.vocabulary = set()
        pass

    def resolve_leaf_brackets(self, TOP: str):
        NEXT = TOP
        self.mapper = {}
        # A closed brackets ( ) in between starts 
        # with capital letters only then a space followed by thing not a bracket
        NE_brackets_regex = r"\([A-Z]+\s[^\()]+\)"
        matches = re.findall(NE_brackets_regex, TOP)

        for id, bracket in enumerate(matches):
            NEXT = NEXT.replace(bracket, f"/{id}")
            # split only the first space (TOPPING x y z) => entity = TOPPING, text = x y z
            entity, text = bracket[1: -1].split(" ", 1) 

            self.mapper[f"/{id}"] = (entity, text)
        
        return NEXT
    
    def get_keyword_brackets(self, TOP: str, keyword: str):
        matches = re.findall(fr"\({keyword}\s[^\(\)]+\)", TOP)
        group = []
        for match in matches:
            _, text = match[1:-1].split(" ", 1)
            sentence, ent = [], []
            for item in text.split(" "):
                if item in self.mapper:
                    words : list = self.mapper[item][1].split(" ")
                    words.remove("")

                    sentence.extend(words)
                    ent.extend([ self.mapper[item][0] ] * len(words))

                    if keyword == "NOT":
                        self.mapper[item] = ( f"NOT_{self.mapper[item][0]}", self.mapper[item][1] )
                else:
                    sentence.append(item)
                    ent.append("NONE")
            group.append((sentence, ent))
            TOP = TOP.replace(match, text)
        return TOP, group

    def formulate_test_case(self, TOP):
        self.x = []
        self.y = []

        for token in TOP.split(" "):
            if token == "": continue

            if token in self.mapper:
                entity, text = self.mapper[token]
                words = text.split(" ")
                words.remove("")

                for word in words:
                    self.vocabulary.add(word)

                self.x.extend(words)
                self.y.extend([entity] * len(words))
            else:
                self.x.append(token)
                self.vocabulary.add(token)
                
                if token in PIZZA_WORDS:
                    self.y.append("PIZZA")
                else:
                    self.y.append("NONE")

    def preprocess(self, TOP):
        TOP                     = self.normalizer.normalize(TOP) ## PREPROCESSING STEP
        TOP                     = self.resolve_leaf_brackets(TOP)
        TOP, complex_toppings   = self.get_keyword_brackets(TOP, keyword="COMPLEXTOPPING")
        TOP, not_groups         = self.get_keyword_brackets(TOP, keyword="NOT")
        TOP, pizza_orders       = self.get_keyword_brackets(TOP, keyword="PIZZAORDER")
        TOP, drink_orders       = self.get_keyword_brackets(TOP, keyword="DRINKORDER")
        TOP, _                  = self.get_keyword_brackets(TOP, keyword="ORDER")
        TOP                     = self.normalizer.reorganize_spaces(TOP) ## PREPROCESSING STEP
        self.formulate_test_case(TOP)

    def extract(self, object, prefix):
        SRC, TOP = object[f"{prefix}.SRC"], object[f"{prefix}.TOP"]
        du.preprocess(TOP)

du = DatasetUtils()
x, y = [], []

# preprocessing training data 
# results is saved in database/(x, y)_train.txt
done = 0
print("Processing training data.... estimated 30mins")
with open("dataset/PIZZA_train.json") as file:
    for line in file:
        obj = json.loads(line.strip())
        du.extract(obj, "train")
        x.append(du.x)
        y.append(du.y)
        done += 1
        print(done, end='\r')
    file.close()
    print()

print("Processing Finished.... Writing Results")
with open("database/x_train.txt", "w") as xfile:
    with open("database/y_train.txt", "w") as yfile:
        for words, entities in zip(x, y):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close()

x, y = [], []
# preprocessing dev data 
# results is saved in database/(x, y)_dev.txt
done = 0
print("Processing dev data.... estimated 30mins")
with open("dataset/PIZZA_dev.json") as file:
    for line in file:
        obj = json.loads(line.strip())
        du.extract(obj, "dev")
        x.append(du.x)
        y.append(du.y)
        done += 1
        print(done, end='\r')
    file.close()


print("Processing Finished.... Writing Results")
with open("database/x_dev.txt", "w") as xfile:
    with open("database/y_dev.txt", "w") as yfile:
        for words, entities in zip(x, y):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close()

print("Writing Vocabulary")
with open("database/vocabulary.txt", "w") as vfile:
    vfile.write("\n".join(du.vocabulary))
    vfile.close()    

print("THANK YOU FOR USING MOA PREPROCESSOR")







# TOP = "(ORDER i want (PIZZAORDER (NUMBER 100 ) pizza with (TOPPING sausage ) (TOPPING bacon ) and no (NOT (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING cheese ) ) ) ) )"
# du.preprocess(TOP)

# print(du.x)
# print(du.y)