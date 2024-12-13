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
    # statistics 
    statistics          : dict 
    def __init__(self):
        self.normalizer = Normalizer()
        self.vocabulary = {}
        self.statistics = {}
        self.debug = 0
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
        x = []
        y = []

        self.debug += 1

        for token in TOP.split(" "):
            if token == "": continue

            if token in self.mapper:
                entity, text = self.mapper[token]
                if entity.startswith("NOT_NOT_"):   entity = entity[len("NOT_NOT_"):]
                if entity == "NOT_QUANTITY":        entity = "QUANTITY"

                words = text.split(" "); words.remove("")

                for word in words:
                    if word not in self.vocabulary: self.vocabulary[word] = 0
                    self.vocabulary[word] += 1

                x.extend(words)
                y.extend([entity] * len(words))
            else:
                entity = "NONE"
                x.append(token)
                
                if token not in self.vocabulary: self.vocabulary[token] = 0
                self.vocabulary[token] += 1
                
                if token in PIZZA_WORDS:
                    y.append("PIZZA")
                else:
                    y.append("NONE")

        normalized = self.normalizer.normalize(" ".join(x), lemmatize=False)
        self.x = []
        self.y = []
        self.tags = []

        it = 0
        norm_it = 0
        while norm_it < len(normalized):
            word, tag = normalized[norm_it]
            # this will be used for words like don't it will be seperated to 2 different words  
            try:
                while not x[it].startswith(word): it += 1 
            except:
                print(f"Error: {self.debug} not found in x")
                quit()
            length = len( word_tokenize(x[it]) ) 

            while length > 0:
                self.x.append( (word, tag) )
                self.y.append(y[it])

                norm_it += 1
                if norm_it == len(normalized): break
                word, tag = normalized[norm_it]

                length -= 1


        print(self.x)

        for i, (word, tag) in  enumerate(self.x):
            resolved = POS_rules(word, tag)
            self.x[i] = resolved
            self.tags.append(tag)

        print(self.x)
        print(self.y)
            

    def preprocess(self, TOP):
        # TOP                     = self.normalizer.normalize(TOP) ## PREPROCESSING STEP
        TOP                     = self.normalizer.reorganize_spaces(TOP)
        TOP                     = self.resolve_leaf_brackets(TOP)
        TOP, complex_groups     = self.get_keyword_brackets(TOP, keyword="COMPLEXTOPPING")
        TOP, not_groups         = self.get_keyword_brackets(TOP, keyword="NOT")
        TOP, pizza_orders       = self.get_keyword_brackets(TOP, keyword="PIZZAORDER")
        TOP, drink_orders       = self.get_keyword_brackets(TOP, keyword="DRINKORDER")
        TOP, _                  = self.get_keyword_brackets(TOP, keyword="ORDER")
        TOP                     = self.normalizer.reorganize_spaces(TOP) ## PREPROCESSING STEP
        self.formulate_test_case(TOP)

    def extract(self, object, prefix):
        SRC, TOP = object[f"{prefix}.SRC"], object[f"{prefix}.TOP"]
        du.preprocess(TOP)

# POS part of speech tagging

TOP = "(ORDER (PIZZAORDER (NUMBER five ) (SIZE medium ) pizzas with (TOPPING tomatoes ) and (TOPPING ham ) ) )"
TOP = "(ORDER can i have (PIZZAORDER (NUMBER one ) pizza without (NOT (TOPPING pineapple ) ) and with (TOPPING olives ) (STYLE thin crust ) ) please )"
TOP = "(ORDER can i have (PIZZAORDER (NUMBER one ) pizza avoid adding any (NOT (TOPPING pineapple ) ) and with (TOPPING olives ) (STYLE thin crust ) ) please )"
TOP = "(ORDER (PIZZAORDER (SIZE large ) pie with (TOPPING green pepper ) and with (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING peperonni ) ) ) )"

SRC = "MD RB VB a medium pizza with mushrooms and chicken and please avoid sausage"
TOP = "NONE NONE NONE NUMBER \2 pizza with \3 and \4 and please avoid \5"
du = NERFormatter()
# du.preprocess(TOP)
# quit()

x = []
y = []

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

        print(du.x)
        print(du.y)
        print(du.tags)
        print("=================================================================")

        done += 1
        if done % 1000 == 0:
            quit()
        print(done, end='\r')
    file.close()
    print()

with open("database/statistics/stats.txt", "w") as file:
    for word, dict in du.statistics.items():
        for entity, freq in dict.items():
            file.write(f"{word},{entity},{freq}")
            file.write("\n")
    file.close()

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