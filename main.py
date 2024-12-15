import pickle
import torch
import os
import re
from classes import *
from preprocessor import Normalizer
normalizer = Normalizer


def is_num(x):
    if normalizer.replace_numbers(x).endswith("NUM"):
        return True
    else:
        return False


def load_model(folder_path: str):
    modelFile = open(f"{folder_path}/model", "rb")
    word2idxFile = open(f"{folder_path}/word2idx", "rb")
    idx2labelFile = open(f"{folder_path}/label2idx", "rb")

    model = pickle.load(modelFile)
    word2idx = pickle.load(word2idxFile)
    idx2label = pickle.load(idx2labelFile)

    modelFile.close()
    word2idxFile.close()
    idx2labelFile.close()

    return model, word2idx, idx2label

def preprocess_sentence(sentence, word2idx, max_len=50, preprocess=True):
    preprocessed = sentence
    if preprocess:
        normalizer = Normalizer()
        preprocessed = normalizer.remove_punctuations(sentence)
        preprocessed = normalizer.replace_numbers_and_keep(preprocessed)
        preprocessed = normalizer.reorganize_spaces(preprocessed)
        preprocessed = preprocessed.lower()
    
    modified_sentence = restructure_model_input(preprocessed)
    words = modified_sentence.split()
    indices = [word2idx.get(word, word2idx['<UNK>']) for word in words] 
    indices = indices[:max_len] + [0] * (max_len - len(indices))
    return preprocessed, torch.tensor([indices])

def restructure_model_input(sentence):
    modified_sentence = []
    for word in sentence.split():
        if word.startswith("lg_num") or word.startswith("sm_num"):
            modified_sentence.append(word[: len("lg_num")])
        else:
            modified_sentence.append(word)
    return " ".join(modified_sentence)

def predict_labels(model, sentence, word2idx, idx2label, device, max_len=50, preprocess=True):
    preprocessed, input_tensor = preprocess_sentence(sentence, word2idx, max_len, preprocess=preprocess)
    input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)  # Get logits
        predictions = output.argmax(dim=-1).squeeze(0)  # Get label indices
    return preprocessed, [idx2label[idx.item()] for idx in predictions if idx.item() in idx2label]

def get_pizza_drink_orders(SRC, order_boundaries: list):
    PIZZA_ORDERS = []
    DRINK_ORDERS = []
    words = SRC.split()
    l, r = 0, -1
    found_end = True
    overlapped = False
    for i in range(len(order_boundaries)):
        if order_boundaries[i].startswith("B_"):
            if not found_end: 
                r = i - 1
                found_end = True
                overlapped = True
            else:
                l = i
                found_end = False
        elif order_boundaries[i].startswith("E_"):
            r = i
            found_end = True
        
        if l <= r:
            sentence = " ".join(words[l : r + 1])
            orderType = order_boundaries[l][len("B_") : ]
            if orderType == "PIZZAORDER":
                PIZZA_ORDERS.append(sentence)
            else:
                DRINK_ORDERS.append(sentence)   
            r = l - 1
            if overlapped:
                l = i
                overlapped = False
    
    return PIZZA_ORDERS, DRINK_ORDERS

def structure_pizza_orders(PIZZA_ORDERS):
    global pizza_model, pizza_word2idx, pizza_idx2label
    STRUCTURED_PIZZA_ORDERS = []

    for order in PIZZA_ORDERS:
        preprocessed, labels = predict_labels(pizza_model, order, pizza_word2idx, pizza_idx2label, "cpu", preprocess=False)
        NUMBER = None
        STYLE = None
        ALL_TOPPINGS = []
        SIZE = None

        def singleLabelEntry(words, labels, token):
            ANS = None
            start_of_token = f"B_{token}"
            end_of_token = f"E_{token}"
            if end_of_token in labels:
                if start_of_token in labels:
                    ANS = " ".join(words[labels.index(start_of_token) : labels.index(end_of_token) + 1])
                else:
                    ANS = words[labels.index(end_of_token)]

                return ANS if not (ANS.startswith("sm_num_") or ANS.startswith("lg_num_")) else ANS[len("lg_num_") : ] 
            return ANS

        def constructToppings(words, labels_, NOT=False):
            start_of_token  = "B_TOPPING"   if not NOT else "B_NOT_TOPPING"
            end_of_token    = "E_TOPPING"   if not NOT else "E_NOT_TOPPING"

            quantity_start  = "B_QUANTITY"  if not NOT else "B_NOT_QUANTITY"
            quantity_end    = "E_QUANTITY"  if not NOT else "E_NOT_QUANTITY"
            while end_of_token in labels_:
                r = labels_.index(end_of_token)
                l = labels_.index(start_of_token)       if start_of_token in labels_[ : r] else r 

                qr = labels_.index(quantity_end)        if quantity_end in labels_[ : l ] else None
                ql = labels_.index(quantity_start)      if qr is not None and quantity_start in labels_[:qr] else qr

                ALL_TOPPINGS.append({
                    "Topping": " ".join( words[l: r+1] ),
                    "Quantity": " ".join( words[ ql : qr + 1 ] ) if ql is not None else None,
                    "NOT": NOT
                })
                
                words = words[r + 1 : ]
                labels_ = labels_[r + 1 : ]

        words = preprocessed.split(" ")
        NUMBER = singleLabelEntry(words, labels, "NUMBER")
        STYLE = singleLabelEntry(words, labels, "STYLE")
        STYLE = singleLabelEntry(words, labels, "NOT_STYLE") if STYLE is None else STYLE
        SIZE = singleLabelEntry(words, labels, "SIZE")
        constructToppings(words, labels, False)
        constructToppings(words, labels, True)

        STRUCTURED_PIZZA_ORDERS.append({
            "NUMBER": NUMBER,
            "SIZE": SIZE,
            "STYLE": STYLE,
            "AllTopping": ALL_TOPPINGS
        })
    return STRUCTURED_PIZZA_ORDERS

def structure_drink_orders(DRINK_ORDERS):
    global drink_model, drink_word2idx, drink_idx2label
    STRUCTURED_DRINK_ORDERS = []
    for order in DRINK_ORDERS:
        preprocessed, labels = predict_labels(drink_model, order, drink_word2idx, drink_idx2label, "cpu", preprocess=False)

        NUMBER = None
        VOLUME = None
        CONTAINER_TYPE = None
        DRINK_TYPE = None

        def singleLabelEntry(words, labels, token):
            start_of_token = f"B_{token}"
            end_of_token = f"E_{token}"
            if end_of_token in labels:
                if start_of_token in labels:
                    ANS = " ".join(words[labels.index(start_of_token) : labels.index(end_of_token) + 1])
                else:
                    ANS = words[labels.index(end_of_token)]
                return ANS if not (ANS.startswith("sm_num_") or ANS.startswith("lg_num_")) else ANS[len("lg_num_") : ] 
            return None

        words = preprocessed.split(" ")
        NUMBER = singleLabelEntry(words, labels, "NUMBER")
        
        VOLUME = singleLabelEntry(words, labels, "VOLUME")

        CONTAINER_TYPE = singleLabelEntry(words, labels, "CONTAINERTYPE")

        DRINK_TYPE = singleLabelEntry(words, labels, "DRINKTYPE")

        STRUCTURED_DRINK_ORDERS.append({
            "NUMBER": NUMBER,
            "VOLUME": VOLUME,
            "CONTAINER_TYPE": CONTAINER_TYPE,
            "DRINK_TYPE": DRINK_TYPE
        })
    return STRUCTURED_DRINK_ORDERS


def rephrase_sentences():
    pass

order_model, order_word2idx, order_idx2label = load_model("./pickles/ORDER")
pizza_model, pizza_word2idx, pizza_idx2label = load_model("./pickles/PIZZA")
drink_model, drink_word2idx, drink_idx2label = load_model("./pickles/DRINK")


test_loader = TestsetLoader()

correct = 0
total = 0

while not test_loader.empty():
    sentence, gold = test_loader.fetch_testcase()
    # sentence = "id like one pizza with extra osama, no mahmoud sauce, add two large coke and one moa"
    sentence = sentence.lower()

    preprocessed, labels_ = predict_labels(order_model, sentence, order_word2idx, order_idx2label, "cpu")
    PIZZA_ORDERS, DRINK_ORDERS = get_pizza_drink_orders(preprocessed, labels_)
    STRUCTURED_PIZZA_ORDERS = structure_pizza_orders(PIZZA_ORDERS)
    STRUCTURED_DRINK_ORDERS = structure_drink_orders(DRINK_ORDERS)
    ORDER = {
        "PIZZA_ORDERS": STRUCTURED_PIZZA_ORDERS,
        "DRINK_ORDERS": STRUCTURED_DRINK_ORDERS
    }

    if is_equal(gold, ORDER):
        correct += 1
    # else:
    #     print(sentence)
    #     BeautifulCompareJson(gold, ORDER)
    #     input("Press Enter To Continue")
    #     os.system("cls")


    total += 1

    # print("DOING MACHINE UNLEARNING PLEASE WAIT")
    # print(total, end="\t")

    

print("OK GUYS IT'S HAPPENING STAY FOKIN KALM")
print(f"FROM TOTAL OF {total} ORDERS" )
print(f"YOU GOT {correct} ONES RIGHT" )
print(f"SO YOUR ACCURACY IS {(correct / total) * 100}%" )
if correct / total > 0.7:
    print("YOU CAN REST NOW, SOLIDER")
else:
    print("ReZero")