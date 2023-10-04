import json
import torch
import redis
import os
from dotenv import load_dotenv
from model import ANeuralNet
from nltk_utils import bag_of_words, tokenize
load_dotenv()

# Menentukan device yang akan digunakan untuk melatih model (dalam hal ini CPU).
device = torch.device('cpu')

# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# Koneksi ke Redis
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', None),
    decode_responses=True
)

# Unggah data intents ke Redis
with open('intents.json', 'r') as f:
    intents = json.load(f)
redis_client.set('intents', json.dumps(intents))

# Ambil data intents dari Redis
intents = json.loads(redis_client.get('intents'))

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

# Menyimpan nilai input_size, hidden_size dan output_size dari data yang telah dimuat
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
# Menyimpan semua kata yang ditemukan dan dan tag yang digunakan dalam data training.
all_words = data['all_words']
tags = data['tags']
# Menyimpan state dictionary dari model yang telah dilatih.
model_state = data["model_state"]

# Membuat instance dari model neural network dengan menggunakan nilai yang telah dimuat.
model = ANeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Karen"

# Definisi kelas untuk response dengan data yang berbeda-beda
class ChatResponse:
    def __init__(self, response, image_url=None, coordinates=None):
        self.response = response
        self.image_url = image_url
        self.coordinates = coordinates


# Fungsi untuk menghasilkan respon chatbot
# berdasarkan input teks yang diberikan.
def get_response(msg):
    # Memproses pesan pengguna dengan fungsi
    # tokenize dari modul nltk_utils.
    sentence = tokenize(msg)
    # Mengubah pesan pengguna menjadi vektor bag-of-words
    # dengan menggunakan fungsi bag_of_words dari modul nltk_utils.
    X = bag_of_words(sentence, all_words)
    # Mengubah vektor bag-of-words menjadi tensor.
    X = X.reshape(1, X.shape[0])
    # Mengubah vektor X menjadi tensor PyTorch dan
    # memindahkannya ke device yang telah ditentukan sebelumnya.
    X = torch.from_numpy(X).to(device)
    # Melakukan inferensi pada model dengan input X.
    print(X)

    # Mengambil posisi (indeks) vektor yang sesuai dengan kata-kata dalam inputan
    positions = [idx for idx, value in enumerate(X[0]) if value == 1]

    # Cetak posisi (indeks) vektor yang sesuai
    print("Posisi (indeks) yang sesuai dengan kata-kata dalam inputan:", positions)

    # Mencetak vektor X yang sesuai dengan kata-kata dalam inputan
    for idx, value in enumerate(X[0]):
        if value == 1:
            print(f"Kata '{all_words[idx]}' memiliki nilai {value}")

    output = model(X)
    print(output)
    # Mencari index dengan nilai terbesar pada output.
    max_value, predicted = torch.max(output, dim=1)
    # print(predicted)
    print(max_value)
    print(predicted.item())
    # Menentukan tag yang sesuai dengan index predicted.
    tag = tags[predicted.item()]
    print(tag)
    # Periksa apakah vektor bag-of-words tidak mengandung nilai 1.0
    if 1.0 not in X[0]:
        return ChatResponse("Maaf saya tidak mengerti. Mohon berikan pertanyaan lain.")
    # Mencari respons dari intents.json berdasarkan tag yang sesuai.
    for intent in intents['intents']:
        if tag == intent["tag"]:
            responses = intent['responses']
            response = responses[0]
            image_url = intent.get('image_url')
            coordinates = intent.get('coordinates')
            return ChatResponse(response, image_url, coordinates)

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        chat_response = get_response(sentence)
        print("Bot:", chat_response.response)
        if chat_response.image_url:
            print("Image URL:", chat_response.image_url)
        if chat_response.coordinates:
            print("Coordinates:", chat_response.coordinates)