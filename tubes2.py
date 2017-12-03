import os, string, nltk, csv, numpy, re, pandas, collections
import xml.dom.minidom
import xml.etree.ElementTree as ET

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from string import digits

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

    with open('dataset/topic/Test101.txt') as listTraining:
        # OPEN TOPIC FILE TO GET ALL TRAINING FILE NAME
        readTestFile = numpy.genfromtxt(listTraining, delimiter=" ", dtype=str)


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

    tmpTesting = []
    for testingFile in readTestFile:
        parseTraining = xml.dom.minidom.parse('dataset/Test101/' + testingFile[1] + '.xml')
        element = parseTraining.documentElement

        # GET TEXT FROM INSIDE OF <TEXT> AND <P> TAG
        getText = element.getElementsByTagName('text')[0]
        paragraphs = getParagraph(getText)

        # PUSH EVERY WORD FROM 1 DOCUMENT TO ARRAY
        words = ''
        for word in paragraphs:
            words = words + ' ' + word.childNodes[0].nodeValue
        tmpTesting.append([str(trainingFile[1]), str(trainingFile[2]), words])
    print tmpTesting
    # print 'Saving to training.csv'
    # CREATE NEW RAW CSV TRAINING FILE FROM NUMPY TMPTRAINING
    arrayTraining = numpy.asarray(tmpTraining)
    numpy.savetxt('training.csv', arrayTraining, fmt='%1s', delimiter='>>')

    arrayTesting = numpy.asarray(tmpTesting)
    numpy.savetxt('testing.csv', arrayTesting, fmt='%1s', delimiter='>>')


def preprocessing():
    # EXTRACTION
    with open('training.csv') as raw:
        rawsTraining = numpy.genfromtxt(raw, delimiter='>>', dtype=str)

    with open('testing.csv') as raw:
        rawsTesting = numpy.genfromtxt(raw, delimiter='>>', dtype=str)

    print 'Starting Preprocessing'

    # STOPWORD REMOVAL
    wordnetLemitizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))

    wordCollectionTraining = []
    for idx in range(0, len(rawsTraining)):
        tokenized = word_tokenize(re.sub(r'''[/.!$%^&*()?'`",:;|0-9+-=]''', '', rawsTraining[idx][2]).lower())
        # print tokenized

        # wordTokenized = wordnetLemitizer.tokenize(tokenized)
        filtered = [token for token in tokenized if not token in stopWords]
        for filteredWord in filtered:
            if filteredWord not in wordCollectionTraining:
                wordCollectionTraining.append(filteredWord)

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

        rawsTraining[idx][2] = ''
        for lemmatizing in lemmatized:
            rawsTraining[idx][2] = rawsTraining[idx][2] + ' ' + lemmatizing


    wordCollectionTesting = []
    for idx in range(0, len(rawsTesting)):
        tokenized = word_tokenize(re.sub(r'''[/.!$%^&*()?'`",:;|0-9+-=]''', '', rawsTesting[idx][2]).lower())
        # print tokenized

        # wordTokenized = wordnetLemitizer.tokenize(tokenized)
        filtered = [token for token in tokenized if not token in stopWords]
        for filteredWord in filtered:
            if filteredWord not in wordCollectionTesting:
                wordCollectionTesting.append(filteredWord)

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

        rawsTesting[idx][2] = ''
        for lemmatizing in lemmatized:
            rawsTesting[idx][2] = rawsTesting[idx][2] + ' ' + lemmatizing

    print 'Saving to preprocess file'

    arrayPreprocess = numpy.asarray(rawsTraining)
    numpy.savetxt('preprocessTraining.csv', arrayPreprocess, fmt='%1s', delimiter='>>')

    arrayPreprocess = numpy.asarray(rawsTesting)
    numpy.savetxt('preprocessTesting.csv', arrayPreprocess, fmt='%1s', delimiter='>>')




def classification():
    with open('preprocessTraining.csv') as raw:
        praprosesTraining = numpy.genfromtxt(raw, delimiter='>>', dtype=str)

    with open('preprocessTesting.csv') as raw:
        praprosesTesting = numpy.genfromtxt(raw, delimiter='>>', dtype=str)
    print 'Starting Classification'

    # =============================================
    getWordsTraining = []
    labelTraining = []
    for idx in range(0, len(praprosesTraining)):
        getWordsTraining.append(praprosesTraining[idx][2])
        labelTraining.append(praprosesTraining[idx][1])
    # labelTraining.sort()

    splitWordsTraining = []
    for idx in range(0, len(getWordsTraining)):
        splited = getWordsTraining[idx].split()
        splitWordsTraining.append(splited)
    # print splitWords

    tfIdfTraining = TfidfVectorizer(tokenizer=lambda x:x,min_df=4,preprocessor=lambda x: x,lowercase=False)
    transformTfIdfTraining = tfIdfTraining.fit_transform(splitWordsTraining)
    classifier = MultinomialNB().fit(transformTfIdfTraining, labelTraining)


    getWordsTesting = []
    labelTesting = []
    for idx in range(0, len(praprosesTesting)):
        getWordsTesting.append(praprosesTesting[idx][2])
        # labelTesting.append(praprosesTesting[idx][1])
        print praprosesTesting[idx][1]
    # print labelTesting
    # labelTesting.sort()

    splitWordsTesting = []
    for idx in range(0, len(getWordsTesting)):
        splited = getWordsTesting[idx].split()
        splitWordsTesting.append(splited)

    # tfIdfTesting = TfidfVectorizer(tokenizer=lambda x:x,min_df=4,preprocessor=lambda x: x,lowercase=False)
    transformTfIdfTesting = tfIdfTraining.transform(splitWordsTesting)
    # classifier = MultinomialNB().fit(transformTfIdfTesting, labelTesting)

    prediction = classifier.predict(transformTfIdfTesting)
    correct, truePositif, trueNegative, falsePositif, falseNegative = 0, 0, 0, 0, 0

    for idx in range(len(labelTesting)):
        if labelTesting[idx] == prediction[idx]:
            correct += 1
        
        if labelTesting[idx] == 0 and prediction[idx] == 0 : truePositif+=1
        elif labelTesting[idx] == 1 and prediction[idx] == 0 : falsePositif+=1
        elif labelTesting[idx] == 0 and prediction[idx] == 1 : falseNegative+=1
        elif labelTesting[idx] == 1 and prediction[idx] == 1 : trueNegative+=1

    # accuration  = correct / len(labelTesting)
    # f1Score     = (2 * truePositif) 
    # / (2 * truePositif+falsePositif+falseNegative)

    # print 'Akurasi      = ', labelTraining
    # print 'F1 Score     = ', f1Score

def main():
    getTraining()
    preprocessing()
    classification()
    print 'Done'
main()