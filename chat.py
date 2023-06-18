import random
import json
import torch
import redis
import os
from dotenv import load_dotenv
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# Koneksi ke Memorystore Redis
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', None),
    decode_responses=True
)

# Unggah data intents ke Redis 
# with open('intents.json', 'r') as f:
#     intents = json.load(f)
# redis_client.set('intents', json.dumps(intents))

# Ambil data intents dari Redis
intents = json.loads(redis_client.get('intents'))

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Karen"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                responses = intent['responses']
                image_url = intent.get('image_url')
                response = random.choice(responses)
                return response, image_url


    return "Maaf saya tidak mengerti. Mohon berikan pertanyaan terkait objek wisata di Kabupaten Bandung.", ""

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        response, image_url = get_response(sentence)
        print("Bot:", response)
        if image_url:
            print("Image URL:", image_url)
