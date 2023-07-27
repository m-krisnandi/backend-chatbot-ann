import numpy as np
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import ANeuralNet

# Memuat data intents dari file 'intents.json'
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Menginisialisasi list untuk menyimpan semua kata, tag, dan data pelatihan (pasangan xy)
all_words = []
tags = []
xy = []

# Melakukan pengulangan pada setiap kalimat di dalam intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # Menambahkan ke dalam list tag
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenisasi setiap kata dalam kalimat
        w = tokenize(pattern)
        # Menambahkan ke dalam list kata-kata
        all_words.extend(w)
        # Menambahkan ke dalam pasangan xy
        xy.append((w, tag))

# Melakukan stemming dan menghapus tanda baca
ignore_words = ['?', '.', '!', ',', '"', "'", '`', '(', ')', '-', '[',
                ']', '{', '}', ';', ':', '@', '#', '$', '%', '^', '&',
                '*', '_', '+', '=', '\\', '|', '/', '~', '<', '>', '€']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Menghapus duplikasi dan melakukan pengurutan
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Membuat data pelatihan
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words untuk setiap pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss hanya memerlukan label kelas
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print("X_train: ", X_train[:, 0])
print("y_train: ", y_train)

# Hyperparameter
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

# Mendefinisikan kelas Dataset khusus untuk pelatihan
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Membuat DataLoader untuk data pelatihan
dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

# Memeriksa apakah CUDA tersedia, jika tidak, gunakan CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Menginisialisasi model jaringan saraf
model = ANeuralNet(input_size, hidden_size, output_size).to(device)

# Mendefinisikan fungsi loss dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Menghitung akurasi
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    epoch_loss = 0

# Variabel untuk pelacakan
loss_list = []
accuracy_list = []

# Loop pelatihan
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(words)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss /= len(train_loader)
    accuracy = correct / total * 100

    loss_list.append(epoch_loss)
    accuracy_list.append(accuracy)

    if (epoch + 1) % 100 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Plot grafik untuk perkembangan pelatihan
plt.plot(loss_list, label='Training Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(accuracy_list, label='Training Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Menyimpan model yang telah dilatih beserta informasi terkait ke dalam file
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Pelatihan selesai. Model disimpan di {FILE}')
