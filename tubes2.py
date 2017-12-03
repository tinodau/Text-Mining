import os, string, nltk, csv, numpy, re, pandas, collections
import xml.dom.minidom
import xml.etree.ElementTree as ET

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from string import digits

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = CountVectorizer()

# classifier = nltk.NaiveBayesClassifier.train(train_set)

print 'Read Data'

def getParagraph(paragraphs):
    for paragraph in paragraphs.childNodes:
        if paragraph.localName == 'p':
            yield paragraph


def getTraining():
    # READ ALL XML TRAINING FILE AND PARSE THE ARTICLE CONTENT TO 1 CSV FILE
    print 'extracting data'
    with open('dataset/topic/Training101.txt') as listTraining:
        # OPEN TOPIC FILE TO GET ALL TRAINING FILE NAME
        readTrainingFile = numpy.genfromtxt(listTraining, delimiter=" ", dtype=str)

    # PARSE FILE CONTENT
    # TRAININGFILE[1] = FILENAME INSIDE NUMPY ARRAY
    tmpTraining = []
    for trainingFile in readTrainingFile:
        parseTraining = xml.dom.minidom.parse('dataset/Training101/' + trainingFile[1] + '.xml')
        element = parseTraining.documentElement

        # GET TEXT FROM INSIDE OF <TEXT> AND <P> TAG
        getText = element.getElementsByTagName('text')[0]
        paragraphs = getParagraph(getText)

        # PUSH EVERY WORD FROM 1 DOCUMENT TO ARRAY
        words = ''
        for word in paragraphs:
            words = words + ' ' + word.childNodes[0].nodeValue
        tmpTraining.append([str(trainingFile[1]), str(trainingFile[2]), words])
    print 'Saving to training.csv'
    # CREATE NEW RAW CSV TRAINING FILE FROM NUMPY TMPTRAINING
    arrayTraining = numpy.asarray(tmpTraining)
    numpy.savetxt('training.csv', arrayTraining, fmt='%1s', delimiter='>>')


def preprocessing():
    # EXTRACTION
    with open('training.csv') as raw:
        raws = numpy.genfromtxt(raw, delimiter='>>', dtype=str)

    print 'Starting Preprocessing'

    # STOPWORD REMOVAL
    wordnetLemitizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    wordCollection = []
    for idx in range(0, len(raws)):
        tokenized = word_tokenize(re.sub(r'''[/.!$%^&*()?'`",:;|0-9+-=]''', '', raws[idx][2]).lower())
        # print tokenized

        # wordTokenized = wordnetLemitizer.tokenize(tokenized)
        filtered = [token for token in tokenized if not token in stopWords]
        for filteredWord in filtered:
            if filteredWord not in wordCollection:
                wordCollection.append(filteredWord)

        # LEMMATIZE
        lemmatized = []
        for idx2 in filtered:
            lem = wordnetLemitizer.lemmatize(idx2)
            lem = wordnetLemitizer.lemmatize(lem, wordnet.ADJ)
            lem = wordnetLemitizer.lemmatize(lem, wordnet.ADV)
            lem = wordnetLemitizer.lemmatize(lem, wordnet.NOUN)
            lem = wordnetLemitizer.lemmatize(lem, wordnet.VERB)
            lemmatized.append(lem)

        # STEMMING
        # stemmed = []
        # for idx2 in filtered:
        #     stem = PorterStemmer().stem(idx2)
        #     stemmed.append(stem)
        # print stemmed

        raws[idx][2] = ''
        for lemmatizing in lemmatized:
            raws[idx][2] = raws[idx][2] + ' ' + lemmatizing

    print 'Saving to preprocess.csv'
    arrayPreprocess = numpy.asarray(raws)
    numpy.savetxt('preprocess.csv', arrayPreprocess, fmt='%1s', delimiter='>>')

def classification():
    with open('preprocess.csv') as raw:
        praproses = numpy.genfromtxt(raw, delimiter='>>', dtype=str)
    print 'Starting Classification'

    TFIDF = TfidfVectorizer(tokenizer=lambda x:x,min_df=4,preprocessor=lambda x: x,lowercase=False)



    print 'saving to bagofwords.csv'
    arrayPreprocess = numpy.asarray(bagOfWordsCollection)
    numpy.savetxt('bagofwords.csv', arrayPreprocess, fmt='%1s', delimiter='>>')

def main():
    getTraining()
    preprocessing()
    classification()
    print 'Done'
main()