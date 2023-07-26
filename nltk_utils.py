import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory() 
stemmer = factory.create_stemmer()

indonesian_stopwords = nltk.corpus.stopwords.words('indonesian')

def tokenize(sentence):
    words = nltk.word_tokenize(sentence)
    words = [w for w in words if w not in indonesian_stopwords]
    return words

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["info", "wisata", "kabupaten", "bandung"]
    words = ["tolong", "berikan", "info", "wisata", "kabupaten", "bandung", "dong"]
    bag = [0, 0, 1, 1, 1, 1, 0]
    """

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

# sentence = ["info", "wisata", "kabupaten", "bandung"]
# words = ["tolong", "berikan", "info", "wisata", "kabupaten", "bandung", "dong"]
# bag = bag_of_words(sentence, words)
# print(bag)


