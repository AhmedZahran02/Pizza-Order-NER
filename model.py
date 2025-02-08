from libraries import *
from classes import TestsetLoader, JsonUtils
from preprocessor import Normalizer

stemmer = PorterStemmer() 

ENTITY_RECOGNIZER_PATH = "models/active/EntityRecognizer.pth"
ORDER_RECOGNIZER_PATH = "models/active/OrderRecoginzer.pth"

LABELER_DATA_PATH = "database/labeler"
GROUPER_DATA_PATH = "database/grouper"
MAX_LEN = 50

SHOW_ERRORS = False

class PizzaSemanticParser:
    def __init__(self):
        self.max_len = MAX_LEN
        self.DEBUG_ERRORS = False
        self.DEBUG_OUTPUT = False
        self.load_models()
    
    # cheese => E_TOPPING
    # cheese burger => B_TOPPING E_TOPPING
    def singleLabelEntry(self, words, labels, token):
        ANS = None
        start_of_token = f"B_{token}"
        end_of_token = f"E_{token}"
        if end_of_token in labels:
            if start_of_token in labels:
                ANS = " ".join(words[labels.index(start_of_token) : labels.index(end_of_token) + 1])
                ANS = re.sub(r"\s(\-)\s", r"-", ANS)
            else:
                ANS = words[labels.index(end_of_token)]

            return ANS if not (ANS.startswith("sm_num_") or ANS.startswith("lg_num_")) else ANS[len("lg_num_") : ] 
        return ANS
    
    def constructToppings(self, words, labels_, NOT=False):
        start_of_token  = "B_TOPPING"   if not NOT else "B_NOT_TOPPING"
        end_of_token    = "E_TOPPING"   if not NOT else "E_NOT_TOPPING"
        quantity_start  = "B_QUANTITY"  if not NOT else "B_NOT_QUANTITY"
        quantity_end    = "E_QUANTITY"  if not NOT else "E_NOT_QUANTITY"
        ALL_TOPPINGS = []
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
        return ALL_TOPPINGS

    def load_models(self):
        def load_single_model(model_folder):
            modelFile       = open(f"{model_folder}/model", "rb")
            word2idxFile    = open(f"{model_folder}/word2idx", "rb")
            idx2labelFile   = open(f"{model_folder}/label2idx", "rb")
            model           = load(modelFile)
            word2idx        = load(word2idxFile)
            idx2label       = load(idx2labelFile)
            modelFile       .close()
            word2idxFile    .close()
            idx2labelFile   .close()
            return model, word2idx, idx2label
        
        self.order_model, self.order_word2idx, self.order_idx2label = load_single_model("./pickles/ORDER")
        torch.save(self.order_model.state_dict(), 'order_model_weights.pth')

        self.pizza_model, self.pizza_word2idx, self.pizza_idx2label = load_single_model("./pickles/PIZZA")
        torch.save(self.pizza_model.state_dict(), 'pizza_model_weights.pth')

        self.drink_model, self.drink_word2idx, self.drink_idx2label = load_single_model("./pickles/DRINK")
        torch.save(self.drink_model.state_dict(), 'drink_model_weights.pth')

    def restructure_model_input(self, sentence):
        modified_sentence = []
        for word in sentence.split():
            if word.startswith("lg_num") or word.startswith("sm_num"):
                modified_sentence.append(word[: len("lg_num")])
            else:
                modified_sentence.append(word)
        return " ".join(modified_sentence)

    def preprocess_sentence(self, sentence, word2idx, preprocess=True):
        preprocessed = sentence
        if preprocess:
            normalizer = Normalizer()
            preprocessed = normalizer.remove_punctuations(sentence)
            preprocessed = normalizer.replace_numbers_and_keep(preprocessed)
            preprocessed = normalizer.reorganize_spaces(preprocessed)
            preprocessed = preprocessed.lower()
        
        modified_sentence = self.restructure_model_input(preprocessed)
        words = modified_sentence.split()
        indices = [word2idx.get(word, word2idx['<UNK>']) for word in words] 
        indices = indices[:self.max_len] + [0] * (self.max_len - len(indices))
        return preprocessed, torch.tensor([indices])

    def predict_labels(self, sentence, model_name, preprocess=True):
        if model_name == "PIZZA":
            model, idx2label, word2idx = self.pizza_model, self.pizza_idx2label, self.pizza_word2idx
        elif model_name == "DRINK":
            model, idx2label, word2idx = self.drink_model, self.drink_idx2label, self.drink_word2idx
        else:
            model, idx2label, word2idx = self.order_model, self.order_idx2label, self.order_word2idx

        preprocessed, input_tensor = self.preprocess_sentence(sentence, word2idx, preprocess=preprocess)
        input_tensor.to("cpu")
        with torch.no_grad():
            output = model(input_tensor)  # Get logits
            predictions = output.argmax(dim=-1).squeeze(0)  # Get label indices
        return preprocessed, [idx2label[idx.item()] for idx in predictions if idx.item() in idx2label]

    def get_pizza_drink_orders(self, SRC, order_boundaries: list):
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

    def structure_pizza_orders(self, PIZZA_ORDERS):
        STRUCTURED_PIZZA_ORDERS = []

        for order in PIZZA_ORDERS:
            preprocessed, labels = self.predict_labels(order, "PIZZA", preprocess=False)
            
            if self.DEBUG_OUTPUT:
                print("================================= PIZZA MODEL OUTPUTS =================================")
                print(preprocessed)
                print(labels)
            words = preprocessed.split(" ")

            NUMBER              = self.singleLabelEntry(words, labels, "NUMBER")

            style_text          = self.singleLabelEntry(words, labels, "STYLE")
            STYLE               = [{"TYPE": style_text, "NOT": False}] if style_text is not None else None

            style_text          = self.singleLabelEntry(words, labels, "NOT_STYLE")
            STYLE               = [{"TYPE": style_text, "NOT": True}] if STYLE is None and style_text is not None else STYLE

            SIZE                = self.singleLabelEntry(words, labels, "SIZE")
            
            ALL_TOPPINGS        = self.constructToppings(words, labels, False)
            ALL_NOT_TOPPINGS    = self.constructToppings(words, labels, True)

            STRUCTURED_PIZZA_ORDERS.append({
                "NUMBER": NUMBER,
                "SIZE": SIZE,
                "STYLE": STYLE,
                "AllTopping": ALL_TOPPINGS + ALL_NOT_TOPPINGS
            })
        return STRUCTURED_PIZZA_ORDERS

    def structure_drink_orders(self, DRINK_ORDERS):
        STRUCTURED_DRINK_ORDERS = []
        for order in DRINK_ORDERS:
            preprocessed, labels = self.predict_labels(order, "DRINK", preprocess=False)
            if self.DEBUG_OUTPUT:
                print("================================= DRINK MODEL OUTPUTS =================================")
                print(preprocessed)
                print(labels)
            words = preprocessed.split(" ")

            NUMBER          = self.singleLabelEntry(words, labels, "NUMBER")
            VOLUME          = self.singleLabelEntry(words, labels, "VOLUME")
            SIZE            = self.singleLabelEntry(words, labels, "SIZE")
            CONTAINER_TYPE  = self.singleLabelEntry(words, labels, "CONTAINERTYPE")
            DRINK_TYPE      = self.singleLabelEntry(words, labels, "DRINKTYPE")

            STRUCTURED_DRINK_ORDERS.append({
                "NUMBER": NUMBER,
                "SIZE": SIZE,
                "VOLUME": VOLUME,
                "CONTAINERTYPE": CONTAINER_TYPE,
                "DRINKTYPE": DRINK_TYPE
            })
        return STRUCTURED_DRINK_ORDERS

    def predict(self, sentence: str):
        preprocessed, labels_       = self.predict_labels(sentence, "ORDER", preprocess=True)
        if self.DEBUG_OUTPUT:
            print("================================= ORDER MODEL OUTPUTS =================================")
            print(preprocessed)
            print(labels_)
        PIZZA_ORDERS, DRINK_ORDERS  = self.get_pizza_drink_orders(preprocessed, labels_)
        STRUCTURED_PIZZA_ORDERS     = self.structure_pizza_orders(PIZZA_ORDERS)
        STRUCTURED_DRINK_ORDERS     = self.structure_drink_orders(DRINK_ORDERS)
        ORDER = {
        "ORDER": 
            {
                "PIZZAORDER": STRUCTURED_PIZZA_ORDERS,
                "DRINKORDER": STRUCTURED_DRINK_ORDERS
            }
        }
        return ORDER
    
    def evaluate(self, DEBUG_ERRORS=False, DEBUG_OUTPUT=False):
        test_loader = TestsetLoader()
        correct = 0
        finished = 0

        self.DEBUG_ERRORS = DEBUG_ERRORS
        self.DEBUG_OUTPUT = DEBUG_OUTPUT
        
        output_file = open("evaluation/output_dev.json", "w+")
        with tqdm(total=test_loader.count(), desc="Manual Progress") as pbar:
            while not test_loader.empty():
                sentence, gold = test_loader.fetch_testcase()
                sentence = sentence.lower()
                prediction = self.predict(sentence)

                if JsonUtils.is_equal(gold, prediction):
                    correct += 1
                    if DEBUG_OUTPUT:
                        print(sentence)
                        JsonUtils.compare(gold, prediction)
                        input("Press Enter To Continue")
                        os.system("cls")
                elif DEBUG_ERRORS:
                    print(sentence)
                    JsonUtils.compare(gold, prediction)
                    input("Press Enter To Continue")
                    os.system("cls")
                
                json.dump(prediction, output_file)
                output_file.write("\n")


                finished += 1
                pbar.update((finished / test_loader.count()) * 2)
        output_file.close()
        print("OK GUYS IT'S HAPPENING STAY FOKIN KALM")
        print(f"FROM TOTAL OF {test_loader.count()} ORDERS" )
        print(f"YOU GOT {correct} ONES RIGHT" )
        print(f"SO YOUR ACCURACY IS {(correct / test_loader.count()) * 100}%" )
