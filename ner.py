from var import *
from preprocessor import *
import re
import json

class NERFormatter:
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
        self.vocabulary = {}
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

                if entity.startswith("NOT_NOT_"):   entity = entity[len("NOT_NOT_"):]
                if entity == "NOT_QUANTITY":        entity = "QUANTITY"

                words = text.split(" ")
                words.remove("")

                for word in words:
                    if word not in self.vocabulary: self.vocabulary[word] = 0
                    self.vocabulary[word] += 1

                self.x.extend(words)
                self.y.extend([entity] * len(words))
            else:
                self.x.append(token)
                
                if token not in self.vocabulary: self.vocabulary[token] = 0
                self.vocabulary[token] += 1
                
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

du = NERFormatter()
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
with open("database/labeler/x_train.txt", "w") as xfile:
    with open("database/labeler/y_train.txt", "w") as yfile:
        for words, entities in zip(x, y):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close()


print("Writing Vocabulary")
with open("database/labeler/vocabulary.txt", "w") as vfile:
    for voc, freq in du.vocabulary.items():
        print (voc, freq, sep="\t")
        vfile.write(f"{voc},{freq}\n")
    vfile.close()   

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
with open("database/labeler/x_dev.txt", "w") as xfile:
    with open("database/labeler/y_dev.txt", "w") as yfile:
        for words, entities in zip(x, y):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close() 

print("THANK YOU FOR USING MOA PREPROCESSOR")







# TOP = "(ORDER i want (PIZZAORDER (NUMBER 100 ) pizza with (TOPPING sausage ) (TOPPING bacon ) and no (NOT (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING cheese ) ) ) ) )"
# du.preprocess(TOP)

# print(du.x)
# print(du.y)