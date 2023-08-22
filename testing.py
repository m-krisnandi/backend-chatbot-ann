import torch
import json
import numpy as np

from nltk_utils import bag_of_words, tokenize, stem
from model import ANeuralNet

# Load the trained model
#testing 1
data = torch.load("data.pth")
#testing 2
# data = torch.load("data2.pth")
#testing 3
# data = torch.load("data3.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = ANeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Load the testing data
with open('testing_data.json', 'r') as f:
    testing_data = json.load(f)

# Preprocess the testing data
X_test = []
y_test = []
for test in testing_data:
    pattern_sentence = test['sentence']
    intent_tag = test['tag']
    tokenized_sentence = tokenize(pattern_sentence)
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    X_test.append(bag_of_words(tokenized_sentence, all_words))
    y_test.append(tags.index(intent_tag))

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

# Evaluate the model on the testing data
model.eval()
incorrect_sentences = []  # List to store incorrect sentences
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(len(X_test)):
        inputs = X_test[i].unsqueeze(0)
        labels = y_test[i].unsqueeze(0)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = (predicted == labels).sum().item() / labels.size(0) * 100
        if accuracy < 100:
            incorrect_sentences.append(i + 1)  # Store the sentence number

        tag = testing_data[i]['tag']
        sentence = testing_data[i]['sentence']
        print(f"Tag: {i+1}: {tag}")
        print(f"Sentence {i+1}: {sentence}")
        print(f"Accuracy = {accuracy:.2f}%")
        print("--------------------")

overall_accuracy = correct / total * 100
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

if len(incorrect_sentences) > 0:
    print("\nSentences with accuracy less than 100%:")
    print(incorrect_sentences)