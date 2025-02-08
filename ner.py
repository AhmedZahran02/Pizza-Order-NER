from libraries import *
from preprocessor import Normalizer

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
        self.vocabulary = set()
        self.statistics = {}
        
        self.PIZZA_FILE = open("regenerator/PIZZA.txt", "a")
        self.DRINK_FILE = open("regenerator/DRINK.txt", "a")
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
            
            # for 3/4 of the training data i will replace the words with another words
            # these words are saved in var.py file
            if random.random() <= 0.4:
                if entity == "TOPPING":
                    text = random.sample(TOPPINGS, 1)[0]
                elif entity == "STYLE":
                    text = random.sample(STYLE, 1)[0]
                elif entity == "DRINKTYPE":
                    text = random.sample(DRINK_TYPES, 1)[0]
                elif entity == "QUANTITY":
                    text = random.sample(QUANTITIES, 1)[0]
            
            # with 0.1 i shuffle the word so that it becomes an unknown word
            if random.random() <= 0.1:
                words = text.split()
                wordIdx = random.randint(0, len(words) - 1)
                word = list(words[wordIdx])
                random.shuffle(word)
                words[wordIdx] = "DISCARD_" + "".join(word)
                text = " ".join([word for word in words if word != ""])

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
                    temp_words : list = self.mapper[item][1].split(" ")
                    words = [ word for word in temp_words if word != "" ]
                    
                    sentence.extend(words)
                    ent.extend([ self.mapper[item][0] ] * len(words))

                    if keyword == "NOT":
                        self.mapper[item] = ( f"NOT_{self.mapper[item][0]}", self.mapper[item][1] )
                else:
                    sentence.append(item)
                    ent.append("NONE")
            group.append((text, ent))
            TOP = TOP.replace(match, text)
        return TOP, group

    def formulate_test_case(self, TOP, type):
        x = []
        y = []

        for token in TOP.split(" "):
            if token == "": continue

            if token in self.mapper:
                entity, text = self.mapper[token]

                text = text.lower()
                text = self.normalizer.reorganize_spaces(text)

                if entity.startswith("NOT_NOT_"):   entity = entity[len("NOT_NOT_"):]

                words = [word for word in text.split(" ") if word != ""]
                if "" in words: raise "WTF"

                for i in range(len(words)):
                    word = words[i]
                    if word.startswith("discard_"): 
                        words[i] = word[len("discard_") : ]
                    else:
                        self.vocabulary.add(word)

                x.extend(words)
                
                if len(words) > 1:
                    y.append(f"B_{entity}")
                    if len(words) - 2 > 0: y.extend([entity] * (len(words) - 2))
                    y.append(f"E_{entity}")
                else:
                    y.append(f"E_{entity}")
            else:
                token = token.lower()
                entity = "NONE"
                x.append(token)
                
                if random.random() > 0.4:
                    self.vocabulary.add(token)
                
                if stemmer.stem(token) in PIZZA_WORDS:
                    y.append("PIZZA")
                else:
                    y.append("NONE")

        if type == "PIZZA":
            self.xPizza.append(x)
            self.yPizza.append(y)
            self.PIZZA_FILE.write( " ".join(x) + "\n" )
        else:
            self.xDrink.append(x)
            self.yDrink.append(y)
            self.DRINK_FILE.write( " ".join(x) + "\n" )

    def preprocess(self, TOP):
        TOP                     = self.normalizer.normalize(TOP) ## PREPROCESSING STEP
        TOP                     = self.resolve_leaf_brackets(TOP)
        TOP, complex_groups     = self.get_keyword_brackets(TOP, keyword="COMPLEXTOPPING")
        TOP, not_groups         = self.get_keyword_brackets(TOP, keyword="NOT")
        TOP, pizza_orders       = self.get_keyword_brackets(TOP, keyword="PIZZAORDER")
        TOP, drink_orders       = self.get_keyword_brackets(TOP, keyword="DRINKORDER")
        TOP, _                  = self.get_keyword_brackets(TOP, keyword="ORDER")
        TOP                     = self.normalizer.reorganize_spaces(TOP) ## PREPROCESSING STEP

        self.xPizza = []
        self.yPizza = []
        self.xDrink = []
        self.yDrink = []

        for order in pizza_orders:
            self.formulate_test_case(order[0], "PIZZA")

        for order in drink_orders:
            self.formulate_test_case(order[0], "DRINK")

    def extract(self, object, prefix):
        SRC, TOP = object[f"{prefix}.SRC"], object[f"{prefix}.TOP"]
        du.preprocess(TOP)

TOP = "(ORDER i need to order (PIZZAORDER (NUMBER one ) (SIZE large ) (STYLE vegetarian ) pizza with (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING banana peppers ) ) ) )"
du = NERFormatter()
# du.preprocess(TOP)
# print(du.xPizza)
# print(du.yPizza)
# print(du.vocabulary)
# quit()

xPizza, yPizza = [], []
xDrink, yDrink = [], []
# preprocessing training data 
# results is saved in database/(x, y)_train.txt
done = 0
print("Processing training data.... estimated 30mins")
with open("dataset/PIZZA_train.json") as file:
    for line in file:
        obj = json.loads(line.strip())
        du.extract(obj, "train")
        xDrink.extend(du.xDrink)
        yDrink.extend(du.yDrink)

        xPizza.extend(du.xPizza)
        yPizza.extend(du.yPizza)
        done += 1
        print(done, end='\r')
    file.close()
    print()

print("Processing Finished.... Writing Results")
with open("database/PizzaLabeler/x_train.txt", "w+") as xfile:
    with open("database/PizzaLabeler/y_train.txt", "w+") as yfile:
        for words, entities in zip(xPizza, yPizza):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close()

    
print("Processing Finished.... Writing Results")
with open("database/DrinkLabeler/x_train.txt", "w") as xfile:
    with open("database/DrinkLabeler/y_train.txt", "w") as yfile:
        for words, entities in zip(xDrink, yDrink):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close()


print("Writing Vocabulary")
with open("database/vocabulary.txt", "w") as vfile:
    for voc in du.vocabulary:
        vfile.write(f"{voc}\n")
    vfile.close()   

xPizza, yPizza = [], []
xDrink, yDrink = [], []
# preprocessing dev data 
# results is saved in database/(x, y)_dev.txt
done = 0
print("Processing dev data.... estimated 30mins")
with open("dataset/PIZZA_dev.json") as file:
    for line in file:
        obj = json.loads(line.strip())
        du.extract(obj, "dev")

        xDrink.extend(du.xDrink)
        yDrink.extend(du.yDrink)

        xPizza.extend(du.xPizza)
        yPizza.extend(du.yPizza)
        done += 1
        print(done, end='\r')
    file.close()


print("Processing Finished.... Writing Results")
with open("database/PizzaLabeler/x_dev.txt", "w") as xfile:
    with open("database/PizzaLabeler/y_dev.txt", "w") as yfile:
        for words, entities in zip(xPizza, yPizza):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close()

    
print("Processing Finished.... Writing Results")
with open("database/DrinkLabeler/x_dev.txt", "w") as xfile:
    with open("database/DrinkLabeler/y_dev.txt", "w") as yfile:
        for words, entities in zip(xDrink, yDrink):
            xfile.write(",".join(words) + "\n")
            yfile.write(",".join(entities) + "\n")
        yfile.close()
    xfile.close()

print("THANK YOU FOR USING MOA PREPROCESSOR")







# TOP = "(ORDER i want (PIZZAORDER (NUMBER 100 ) pizza with (TOPPING sausage ) (TOPPING bacon ) and no (NOT (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING cheese ) ) ) ) )"
# du.preprocess(TOP)

# print(du.x)
# print(du.y)