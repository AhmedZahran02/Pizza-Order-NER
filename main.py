from model import PizzaSemanticParser
from preprocessor import Normalizer
from var import BLACKLIST

import re
import os

# Initialize the semantic parser

NER = PizzaSemanticParser("database/models/labeler_model/", "labeler")
OR = PizzaSemanticParser("database/models/grouper_model/", "grouper")

NER.load_model_weights("models/active/NER.pth")
OR.load_model_weights("models/active/OR.pth")

# input("Press any key to continue")
os.system("cls")
# NER.evaluate_model()
# OR.evaluate_model()
# quit()
# Load your test data

def map_input_to_preprocessed(sentence:str):
    normalizer = Normalizer()
    tokens = sentence.split(" ")
    id = 0
    updated_tokens = []
    for token in tokens:
        no_punc = normalizer.remove_punctuations(token).lower()
        no_punc = normalizer.stem_word(no_punc)

        if no_punc not in BLACKLIST:
            updated_tokens.append(f"/{id}")
            id += 1
        else:
            updated_tokens.append(token)

    return updated_tokens


def get_full_entity_recognition(mapped_sentence, recognized_entities, length):
    full_recognized_entities = ["NONE"] * length

    for i in range(length):
        token = mapped_sentence[i]
        if token.startswith("/"):
            entity_id = int(token[1:])
            full_recognized_entities[i] = recognized_entities[entity_id]

    return full_recognized_entities

def get_orders_boundaries(mapped_sentence: list[str], full_recognized_entities: list[str], recognized_orders: list[str], sentence: str):
    orders = []
    pizza_order = []
    words = sentence.split()

    for i in range(len(words)):
        token = mapped_sentence[i]
        pizza_order.append((words[i], full_recognized_entities[i]))
        if token.startswith("/"):
            entity_id = int(token[1:])
            if recognized_orders[entity_id] == "EOO":
                orders.append(pizza_order)
                pizza_order = []
    return orders

'''
A Valid Pizza Order Should Have The Following
    1- only one number
'''
def is_valid_pizza_order(order):
    has_number = 0
    for _, entity in order:
        # order can't have more than one number field
        has_number += (entity == "NUMBER")
    
    return True
    

# input for this function will be a signle order (that was identified)
def create_pizza_order(order, is_pizza=True):
    i = 0
    topping_object = []

    take_as_quantity = False
    quantity = []
    topping = []
    number = []
    style = []
    size = []

    while i < len(order):
        if order[i][1] == "NUMBER":
            number.append(order[i][0])

        elif order[i][1].endswith("STYLE"):
            style.append((order[i][0], True if order[i][1].startswith("NOT") else False))    
        
        elif order[i][1] == "SIZE":
            size.append(order[i][0])

        elif order[i][1] == "QUANTITY":
            quantity.append(order[i][0])
            take_as_quantity = True

        elif order[i][1].endswith("TOPPING"):
            NOT = order[i][1].startswith("NOT")
            while i < len(order) and (order[i][1].endswith("TOPPING") or order[i][0] in ["-"]): 
                topping.append(order[i][0])
                i += 1
            
            topping_object.append((" ".join(topping), " ".join(quantity), NOT))

            take_as_quantity = False
            topping = []
            quantity = []
            i -= 1

        elif take_as_quantity:
            quantity.append(order[i][0])

        i += 1

    return (
        topping_object,
        number,
        style,
        size
    )

testcases = []
with open("dataset/PIZZA_test.txt") as f:
    for line in f:
        testcases.append(line.strip())

for test in testcases:
    length = len(test.split())
    recognized_entities, preprocessed = NER.predict(test)[:length]
    mapped_sentence = map_input_to_preprocessed(test)
    full_recognized_entities = get_full_entity_recognition(mapped_sentence, recognized_entities, length)
    
    recognized_orders, preprocessed = OR.predict(test)[:length]
    orders = get_orders_boundaries(mapped_sentence, full_recognized_entities, recognized_orders, test)

    print(test)
    print("================================")
    print(" ".join(full_recognized_entities))
    print("================================")
    print(" ".join(recognized_orders))

    for order in orders:
        if not is_valid_pizza_order(order): continue
        json_order = create_pizza_order(order)
        print(order)
        print("================================")
        print(json_order)
    input("Press To Keep Going")
    os.system("cls")
    







