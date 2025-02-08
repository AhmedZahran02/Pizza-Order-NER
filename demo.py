from libraries import *
from model import PizzaSemanticParser
from classes import JsonUtils

# testset_csv = pd.read_csv('database/test_set.csv')
model = PizzaSemanticParser()
while True:
    sentence = input("Please Enter Your Order Sir: \n => ")
    prediction = model.predict(sentence)

    print("Your Order Sentence Is " + sentence)
    print("The Predicted Pizza Mania Order Is:")
    JsonUtils.log(prediction)

    print("\nDo you want to try another order? (y/n): ")
    choice = input().lower()
    if choice != 'y':
        break
    os.system('cls')
