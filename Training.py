from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import xml.etree.cElementTree as ET
import nltk
import re
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

XML_FILE = open("D:\\Python_programs\\NeuroNetwork\\data\\SentiRuEval_rest_markup_train.xml", encoding="utf8")

model = Word2Vec(sentences=None, iter=5)
tree = ET.ElementTree(file=XML_FILE)
root = tree.getroot()
# print(root.tag)

List = []
Train = []
vocabulary = []
for item in root.iterfind("./review/aspects"):
    for unit in item.iterfind("./aspect"):
        # print(unit.attrib['term'])
        List.append(str(unit.attrib['term']).lower())
    vocabulary.append(List)
    List = []


model = Word2Vec(vocabulary, iter=3, min_count=5)
# model.build_vocab(vocabulary)
# model.train(vocabulary)
print(model)
print(model.most_similar('блюда', topn=5))
print(model['блюда'])
print(model['кухня'])
model1 = model['блюда'] + model['кухня']
print("model1 = ")
print(model1)
print("model1/2 = ")
print(model1/2)

List = []
ArrayList = []
Sentence = []
buf = []
stop_words = stopwords.words('russian')
additional_stop_words = [',', '.', '?', '!', '(', ')', '...', '{', '}', '[', ']', 'это', ':', '-', '+', "''"]
stop_words += additional_stop_words
print(stop_words)
for item in root.iterfind("./review/text"):
    buf = word_tokenize(item.text)
    Sentence.append([w.lower() for w in buf if not w in (stop_words)])
    Sentence.clear()
