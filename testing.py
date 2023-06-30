import torch
import json
import numpy as np

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

ignore_words = ['?', '.', '!', ',', '"', '’', '‘', '“', '”', '(', ')', '[', ']', '{', '}']

# Load the trained model
data = torch.load("data.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Load the testing data
with open('testing_data.json', 'r') as f:
    testing_data = json.load(f)

# Preprocess the testing data
X_test = []
y_test = []
accuracies = []  # List to store accuracies for each sentence
for test in testing_data:
    pattern_sentence = test['sentence']
    intent_tag = test['tag']
    tokenized_sentence = tokenize(pattern_sentence)
    tokenized_sentence = [stem(w) for w in tokenized_sentence if w not in ignore_words]
    X_test.append(bag_of_words(tokenized_sentence, all_words))
    y_test.append(tags.index(intent_tag))

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

# Evaluate the model on the testing data
model.eval()
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
        accuracies.append(accuracy)
        
        sentence = testing_data[i]['sentence']
        print(f"Sentence {i+1}: {sentence}")
        print(f"Accuracy = {accuracy:.2f}%")
        print("--------------------")

overall_accuracy = correct / total * 100
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
