import os, string, nltk, csv, numpy
# import string
# import nltk
import xml.etree.ElementTree as ET

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download('stopwords')
# nltk.download('punkt')

# GET ALL XML FILE AND GET ALL TEXT ARTICLE
# THEN CONVERT ALL TEXT TO LOWERCASE
words = []
path = 'dataset/Training101'
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    file = os.path.join(path, filename)
    tree = ET.parse(file)
    root = tree.getroot()

    # LOAD ALL PARAGRAPH IN <TEXT> TAG IN 1 DOCUMENT
    for elements in root.findall('text'):
        for element in elements:
            # tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

            # STOPWORD REMOVAL
            stop_words = set(stopwords.words('english'))

            # TOKENIZATION
            word_tokens = word_tokenize(element.text.lower())

            # APPEND WORD TO ARRAY IF NOT STOPWORD
            filtered_sentence = [w for w in word_tokens if not w in stop_words]

            filtered_sentence = []
            stemming = []
            text = ""

            for w in word_tokens:
                if w not in stop_words:
                    filtered_sentence.append(w)
                    text = text + w

            print filtered_sentence
            for filtered in filtered_sentence:
                words.append(PorterStemmer().stem(filtered))
a = numpy.asarray(words)
numpy.savetxt('coba.csv', a, fmt='%2s', delimiter=',')


# print(word_tokens), "\n \n"
# print(filtered_sentence)
