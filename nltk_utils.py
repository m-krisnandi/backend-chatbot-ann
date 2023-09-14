import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

indonesian_stopwords = nltk.corpus.stopwords.words('indonesian')

def tokenize(sentence):
    words = nltk.word_tokenize(sentence.lower())
    words = [w for w in words if w not in indonesian_stopwords]
    return words

def stem(word):
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, all_words):
    """
    tokenized_sentence = ['beritahu', 'info', 'wisata', 'kawah', 'putih', 'kabupaten', 'bandung', '']
    all_words = ['info', 'wisata', 'kawah', 'putih', 'di', 'kabupaten', 'bandung']
    bag = [1. 1. 1. 1. 0. 1. 1.]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

# Testing Tokenizing
# sentence = ("Beritahu Saya dan Infokan Wisata "
#             "Kawah Putih di Kabupaten Bandung!")
# words = tokenize(sentence)
# print(words)
# Output: ['Beritahu', 'Saya', 'dan', 'Infokan', 'Wisata', 'Kawah', 'Putih', 'di', 'Kabupaten', 'Bandung', '!']

# Testing Case Folding
# word = ['Beritahu', 'Saya', 'dan', 'Infokan', 'Wisata', 'Kawah', 'Putih', 'di', 'Kabupaten', 'Bandung', '!']
# case_folding = [w.lower() for w in word]
# print(case_folding)
# Output: ['beritahu', 'saya', 'dan', 'infokan', 'wisata', 'kawah', 'putih', 'di', 'kabupaten', 'bandung', '!']

# Testing Stopword
# tokenize_word = ['beritahu', 'saya', 'dan', 'infokan', 'wisata', 'kawah', 'putih', 'di', 'kabupaten', 'bandung', '!']
# stopword = [w for w in tokenize_word if w not in indonesian_stopwords]
# print(stopword)
# Output: ['beritahu', 'infokan', 'wisata', 'kawah', 'putih', 'kabupaten', 'bandung', '!']

# Testing Stemming
# words = ['beritahu', 'infokan', 'wisata', 'kawah', 'putih', 'kabupaten', 'bandung', '!']
# stemming = [stemmer.stem(w) for w in words]
# print(stemming)
# Output: ['beritahu', 'info', 'wisata', 'kawah', 'putih', 'kabupaten', 'bandung', '']

# Testing Bag of Words
# tokenized_sentence = ['beritahu', 'info', 'wisata', 'kawah', 'putih', 'kabupaten', 'bandung', '']
# all_words = ['info', 'wisata', 'kawah', 'putih', 'di', 'kabupaten', 'bandung']
# bag = bag_of_words(tokenized_sentence, all_words)
# print(bag)
# Output: [1. 1. 1. 1. 0. 1. 1.]


