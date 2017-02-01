from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import xml.etree.cElementTree as ET
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

XML_FILE = open("SentiRuEval_rest_markup_train.xml", encoding="utf8")
XML_FILE_2 = open("SentiRuEval_rest_markup_test.xml", encoding="utf8")


model = Word2Vec(sentences=None, iter=5)
tree = ET.ElementTree(file=XML_FILE)
root = tree.getroot()


List = []
Train = []
vocabulary = []


List = []

ArrayList = []
Sentence = []
buf = []
stop_words = stopwords.words('russian')
additional_stop_words = [',', '.', '?', '!', '(', ')', '...', '{', '}', '[', ']', 'это', ':', '-', '+', "''", '``']
stop_words += additional_stop_words
print(stop_words)
for item in root.iterfind("./review/text"):
    # print(str(count) + ": " + item.text)
    ArrayList.append(word_tokenize(item.text))
    buf = word_tokenize(item.text)
    Sentence.append([w.lower() for w in buf if not w in (stop_words)])

    sentences = sent_tokenize(item.text)
    for unit in sentences:
        word_tokens = word_tokenize(unit)
        List.append([w.lower() for w in word_tokens if not w in (stop_words)])


model = Word2Vec(List, iter=25, min_count=1)

print(model.most_similar('блюда', topn=10))
print(model)
Texts = []
buf = []
# count = 0
for item in root.iterfind("./review/text"):
    buf = word_tokenize(item.text)
    Texts.append([w.lower() for w in buf if not w in (stop_words)])

count = 0
amounts = []
sum = 0

for text in Texts:
    # print(text)
    length = len(text)
    for word in text:
        try:
            sum = sum + model[word]
        except KeyError:
            length -= 1

    average = sum/length
    sum = 0
    amounts.append(average)
    count += 1

Marks = []
for categories in root.iterfind("./review/categories"):
    # print(categories)
    for category in categories:
        if category.attrib.get('name') == 'Whole':
            Marks.append(category.attrib.get('sentiment'))


X = amounts
Y = []
for sentiment in Marks:
    if sentiment == 'both':
        Y.append(0)
    elif sentiment == 'positive':
        Y.append(1)
    else:
        Y.append(2)

print(Y)

classifier = svm.SVC(kernel='linear', C=100000, gamma=1)
# Используемые параметры можно варьировать
classifier.fit(X, Y)
classifier.score(X, Y)
print(classifier)


Texts = []
count = 0
tree1 = ET.ElementTree(file=XML_FILE_2)
root1 = tree1.getroot()
amounts = []

for item in root1.iterfind("./review/text"):
    buf = word_tokenize(item.text)
    Texts.append([w.lower() for w in buf if not w in (stop_words)])


for text in Texts:
    # print(text)
    length = len(text)
    for word in text:
        try:
            sum = sum + model[word]
        except KeyError:
            length -= 1

    average = sum / length
    sum = 0
    # print(str(type(average)))
    if type(average) == np.ndarray:
        amounts.append(average)
    count += 1

total = count
# print(len(amounts))
# print(count)

# Прогноз
predicted = classifier.predict(amounts)
print(predicted)
count = 0
correct = 0
Rating = []
for categories in root1.iterfind("./review/categories"):
    # print(categories)
    for category in categories:
        if category.attrib.get('name') == 'Whole':
            if predicted[count] == 0 and category.attrib.get('sentiment') == "both":
                Rating.append("both - correct")
                correct += 1
            elif predicted[count] == 1 and category.attrib.get('sentiment') == 'positive':
                Rating.append("positive - correct")
                correct += 1
            elif predicted[count] == 2 and category.attrib.get('sentiment') == 'negative':
                Rating.append("negative - correct")
                correct += 1
            else:
                Rating.append("bad result")
    count += 1

print(Rating)

print("the percentage of correct certain reviews equals "  + str(correct/total*100))
