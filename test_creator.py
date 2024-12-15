from var import *
from preprocessor import *
import re
import json
import random

class TESTGoldOutputGenerator:
    y: dict
    vocabulary: set
    mapper              : dict

    pizza_orders        : dict
    drink_orders        : dict
    def __init__(self):
        self.normalizer = Normalizer()
        self.y = {
            "PIZZA_ORDERS": [],
            "DRINK_ORDERS": []
        }
        pass

    def resolve_leaf_brackets(self, TOP: str):
        NEXT = TOP
        self.mapper = {}

        NE_brackets_regex = r"\([A-Z]+\s[^\()]+\)"
        matches = re.findall(NE_brackets_regex, TOP)

        for id, bracket in enumerate(matches):
            NEXT = NEXT.replace(bracket, f"/{id}")
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
        
        PIZZA_ORDER = {
            "NUMBER": None,
            "SIZE": None,
            "STYLE": None,
            "AllTopping": []
        }

        DRINK_ORDER = {
            "NUMBER": None,
            "VOLUME": None,
            "CONTAINER_TYPE": None,
            "DRINK_TYPE": None
        }

        words = [word for word in TOP.split(" ") if word != '']

        TOPPING_OBJECT = {
            "Topping": None,
            "Quantity": None,
            "NOT": None
        }

        i = 0
        for token in words:
            if token == "": continue

            if token in self.mapper:
                entity, text = self.mapper[token]
                if entity.startswith("NOT_NOT_"):   entity = entity[len("NOT_NOT_"):]

                text = text.lower()
                text = self.normalizer.reorganize_spaces(text)

                if entity.endswith("QUANTITY"):
                    TOPPING_OBJECT["Quantity"] = text

                elif entity.endswith("TOPPING"):
                    TOPPING_OBJECT["Topping"] = text
                    TOPPING_OBJECT["NOT"] = entity.startswith("NOT_")
                    PIZZA_ORDER["AllTopping"].append(TOPPING_OBJECT) 
                    TOPPING_OBJECT = { "Topping": None, "Quantity": None, "NOT": None }

                elif entity.endswith("NUMBER"):
                    PIZZA_ORDER["NUMBER"] = text
                    DRINK_ORDER["NUMBER"] = text

                elif entity.endswith("STYLE"):
                    PIZZA_ORDER["STYLE"] = text

                elif entity.endswith("SIZE"):
                    PIZZA_ORDER["SIZE"] = text

                elif entity.endswith("VOLUME"):
                    DRINK_ORDER["VOLUME"] = text

                elif entity.endswith("CONTAINERTYPE"):
                    DRINK_ORDER["CONTAINER_TYPE"] = text

                elif entity.endswith("DRINKTYPE"):
                    DRINK_ORDER["DRINK_TYPE"] = text

        if type == "PIZZA":
            self.y["PIZZA_ORDERS"].append(PIZZA_ORDER)
        else:
            self.y["DRINK_ORDERS"].append(DRINK_ORDER)
                
                

    def preprocess(self, TOP):
        TOP                     = self.normalizer.reorganize_spaces(TOP)
        TOP                     = self.resolve_leaf_brackets(TOP)
        TOP, _                  = self.get_keyword_brackets(TOP, keyword="COMPLEX_TOPPING")
        TOP, _                  = self.get_keyword_brackets(TOP, keyword="NOT")
        TOP, pizza_orders       = self.get_keyword_brackets(TOP, keyword="PIZZAORDER")
        TOP, drink_orders       = self.get_keyword_brackets(TOP, keyword="DRINKORDER")
        TOP, _                  = self.get_keyword_brackets(TOP, keyword="ORDER")
        TOP                     = self.normalizer.reorganize_spaces(TOP) 

        self.y = {
            "PIZZA_ORDERS": [],
            "DRINK_ORDERS": []
        }

        for order in pizza_orders:
            self.formulate_test_case(order[0], "PIZZA")

        for order in drink_orders:
            self.formulate_test_case(order[0], "DRINK")


# TOP = "(ORDER i need to order (PIZZAORDER (NUMBER one ) (SIZE large ) (STYLE vegetarian ) pizza with (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING banana peppers ) ) ) )"
# du = TESTGoldOutputGenerator()
# du.preprocess(TOP)
# print(du.y)
# quit()




