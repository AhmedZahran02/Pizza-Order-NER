from var import *
from preprocessor import *
from nltk.stem import PorterStemmer

import re
import json

stemmer = PorterStemmer()

class OrderFormatter:
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
        self.mapper = {}

    def resolve_leaf_brackets(self, TOP: str):
        NE_brackets_regex = r"\([A-Z]+\s[^\()]+\)"
        matches = re.findall(NE_brackets_regex, TOP)

        for bracket in matches:
            _, text = bracket[1: -1].split(" ", 1) 
            TOP = TOP.replace(bracket, text)
        
        return TOP
    
    def remove_keywords(self, TOP: str, keywords: list[str]):
        matches = re.findall(fr"\((?:{'|'.join(keywords)})\s[^\(\)]+\)", TOP)
        for match in matches:
            _, text = match[1:-1].split(" ", 1)
            TOP = TOP.replace(match, text)
        return TOP
    
    def include_classes(self, TOP: str, classes: list[str]):
        matches = re.findall(fr"\((?:{'|'.join(classes)})\s[^\(\)]+\)", TOP)
        for id, match in enumerate(matches):
            entity, text = match[1:-1].split(" ", 1)
            self.mapper[f"/{id}"] = (entity, text)
            TOP = TOP.replace(match, f"/{id}")
        return TOP

    def formulate_test_case(self, TOP):
        self.x = []
        self.y = []

        for token in TOP.split(" "):
            if token == "": continue

            if token in self.mapper:
                entity, text = self.mapper[token]
                text = self.normalizer.reorganize_spaces(text)
                words : list = text.split(" ")

                for word in words:
                    self.vocabulary.add(word)

                self.x.extend(words)
                self.y.extend([entity] * (len(words) - 1))
                self.y.append("EOO") # End Of Order
            else:
                self.x.append(token)
                self.vocabulary.add(token)
                self.y.append("NONE")

    def preprocess(self, TOP):
        TOP = self.normalizer.normalize(TOP) ## PREPROCESSING STEP
        TOP = self.resolve_leaf_brackets(TOP)
        TOP = self.remove_keywords(TOP, ["TOPPING", "NUMBER", "QUANTITY", "STYLE", "DRINKTYPE", "CONTAINERTYPE"])
        TOP = self.remove_keywords(TOP, ["COMPLEXTOPPING"])
        TOP = self.remove_keywords(TOP, ["NOT"])
        TOP = self.include_classes(TOP, ["PIZZAORDER", "DRINKORDER"])
        TOP = self.remove_keywords(TOP, ["ORDER"])
        TOP = self.normalizer.reorganize_spaces(TOP) ## PREPROCESSING STEP
        self.formulate_test_case(TOP)

    def extract(self, object, prefix):
        SRC, TOP = object[f"{prefix}.SRC"], object[f"{prefix}.TOP"]
        of.preprocess(TOP)




of = OrderFormatter()
x, y = [], []

# preprocessing training data 
# results is saved in database/grouper/(x, y)_train.txt
done = 0
print("Processing training data.... estimated 30mins")
with open("dataset/PIZZA_train.json") as file:
    for line in file:
        obj = json.loads(line.strip())
        of.extract(obj, "train")
        x.append(of.x)
        y.append(of.y)
        done += 1
        print(done, end='\r')
    file.close()
    print()

print("Processing Finished.... Writing Results")
with open("database/grouper/x_train.txt", "w") as xfile:
    with open("database/grouper/y_train.txt", "w") as yfile:
        for words, entities in zip(x, y):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close()

x, y = [], []
# preprocessing dev data 
# results is saved in database/(x, y)_dev.txt
done = 0
print("Processing dev data.... no estimation cuz i'm lazy to calculate")
with open("dataset/PIZZA_dev.json") as file:
    for line in file:
        obj = json.loads(line.strip())
        of.extract(obj, "dev")
        x.append(of.x)
        y.append(of.y)
        done += 1
        print(done, end='\r')
    file.close()


print("Processing Finished.... Writing Results")
with open("database/grouper/x_dev.txt", "w") as xfile:
    with open("database/grouper/y_dev.txt", "w") as yfile:
        for words, entities in zip(x, y):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close()

print("Writing Vocabulary")
with open("database/grouper/vocabulary.txt", "w") as vfile:
    vfile.write("\n".join(of.vocabulary))
    vfile.close()    

print("THANK YOU FOR USING MOA PREPROCESSOR")


# TOP = "(ORDER i want (PIZZAORDER (NUMBER 100 ) pizza with (TOPPING sausage ) (TOPPING bacon ) and no (NOT (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING cheese ) ) ) ) )"
# du.preprocess(TOP)

# print(du.x)
# print(du.y)