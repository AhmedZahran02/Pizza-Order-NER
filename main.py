from libraries import *
from model import PizzaSemanticParser

testset_csv = pd.read_csv('database/test_set.csv')
model = PizzaSemanticParser()
outputs = []

# model.evaluate(DEBUG_ERRORS=False)
# quit()
for id, order in enumerate(testset_csv["order"]):
    prediction = model.predict(order)
    outputs.append(prediction)
    # print()
    # print(order)
    # BeautifulPrintJson(prediction)
    # print("Press Enter To Continue")
    # input()
    # os.system("cls")


with open("evaluation/output.json", "w") as file:
    for output in outputs:
        json.dump(output, file)
        file.write("\n")