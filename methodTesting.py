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
from sklearn.metrics import classification_report

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

XML_FILE = open("SentiRuEval_rest_markup_train.xml", encoding="utf8")
XML_FILE_2 = open("SentiRuEval_rest_markup_test.xml", encoding="utf8")

# создание и подготовка модели
model = Word2Vec(sentences=None, iter=5)
# открытие xml документов
tree = ET.ElementTree(file=XML_FILE)
root = tree.getroot()

List = []
Train = []
vocabulary = []


List = []

ArrayList = []
Sentence = []
buf = []
# использование стоп-слов из библиотеки stopwords корпуса nltk.corpus
stop_words = stopwords.words('russian')
# добавление дополнительных стоп-слов
additional_stop_words = [',', '.', '?', '!', '(', ')', '...', '{', '}', '[', ']', 'это', ':', '-', '+', "''", '``']
stop_words += additional_stop_words
print(stop_words)
# Берем каждый текст из xml с тренировоынми даными, разбиваем его на предложения, а предложения на слова и записываем этот текст,
# как список списков предложений в общий список текстов (исключая стоп-слова).
for item in root.iterfind("./review/text"):
    sentences = sent_tokenize(item.text)
    for unit in sentences:
        word_tokens = word_tokenize(unit)
        List.append([w.lower() for w in word_tokens if not w in (stop_words)])

        # print(str(count) + ": " + item.text)
        # ArrayList.append(word_tokenize(item.text))
        # buf = word_tokenize(item.text)
        # Sentence.append([w.lower() for w in buf if not w in (stop_words)])

# Обучаем модель
model = Word2Vec(List, iter=25, min_count=1)
# Пример работы полученной модели
print(model.most_similar('блюда', topn=10))
print(model)
Texts = []
buf = []
# count = 0
# Теперь опять проходим по текстам, но уже для того, чтобы собрать список слов каждого предложения.
for item in root.iterfind("./review/text"):
    buf = word_tokenize(item.text)
    Texts.append([w.lower() for w in buf if not w in (stop_words)])

count = 0
amounts = []
sum = 0
# Составляем вектора текстов
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
# Собираем оценки рецензий
for categories in root.iterfind("./review/categories"):
    # print(categories)
    for category in categories:
        if category.attrib.get('name') == 'Whole':
            Marks.append(category.attrib.get('sentiment'))


X = amounts
Y = []
# Сопоставляем полученные вектора оценкам
for sentiment in Marks:
    if sentiment == 'both':
        Y.append(0)
    elif sentiment == 'positive':
        Y.append(1)
    else:
        Y.append(2)

print(Y) # Вектор оценок
# Используем модель опорных векторов
classifier = svm.SVC(kernel='linear', C=100000, gamma=1)
# Используемые параметры можно варьировать
classifier.fit(X, Y)
classifier.score(X, Y)
print(classifier)


Texts = []
count = 0
# Берем тестовые данные
tree1 = ET.ElementTree(file=XML_FILE_2)
root1 = tree1.getroot()
amounts = []

for item in root1.iterfind("./review/text"):
    buf = word_tokenize(item.text)
    Texts.append([w.lower() for w in buf if not w in (stop_words)])

# Составляем векторное представление текстов тестовой выборки
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

total = count # общая сумма текстов тестовых данных
# print(len(amounts))
print(count)

# Прогноз
predicted = classifier.predict(amounts) # Подача векторов текстов модели SVM
print(predicted)
count = 0
correct = 0
Rating = []
# Составление наглядного массива результативности алгоритма
for categories in root1.iterfind("./review/categories"):
    # print(categories)
    for category in categories:
        if category.attrib.get('name') == 'Whole':
            if predicted[count] == 0 and category.attrib.get('sentiment') == "both":
                Rating.append("both - correct")
                correct += 1 # Если отгадано правильно, инкерментируем подсчет количества правильно отгаданных рецензий
            elif predicted[count] == 1 and category.attrib.get('sentiment') == 'positive':
                Rating.append("positive - correct")
                correct += 1 # Если отгадано правильно, инкерментируем подсчет количества правильно отгаданных рецензий
            elif predicted[count] == 2 and category.attrib.get('sentiment') == 'negative':
                Rating.append("negative - correct")
                correct += 1 # Если отгадано правильно, инкерментируем подсчет количества правильно отгаданных рецензий
            else:
                Rating.append("bad result")
    count += 1
print(Rating)

print("the percentage of correct certain reviews equals " + str(correct/total*100)) # Вывод подсчета процента правлиьно отгаданных ооценок рецензий

Marks_Fact = []
for categories in root1.iterfind("./review/categories"):
    for category in categories:
        if category.attrib.get('name') == 'Whole':
            Marks_Fact.append(category.attrib.get('sentiment'))

Y_Fact = []
for sentiment in Marks_Fact:
    if sentiment == 'both':
        Y_Fact.append(0)
    elif sentiment == 'positive':
        Y_Fact.append(1)
    else:
        Y_Fact.append(2)

print("Fact: " + str(Y_Fact))
print()
print("Predicted: " + str(predicted))

target_names = ['both', 'positive', 'negative']
print(classification_report(Y_Fact, predicted, target_names=target_names))