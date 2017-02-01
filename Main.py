from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import xml.etree.cElementTree as ET
import nltk
import re
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

XML_FILE = open("D:\\Python_programs\\NeuroNetwork\\data\\SentiRuEval_rest_markup_train.xml", encoding="utf8")
tree = ET.ElementTree(file=XML_FILE)
root = tree.getroot()
count = 0
List = []
ArrayList = []
stop_words = stopwords.words('russian')
stop_words += ','
stop_words += '.'
stop_words += '?'
stop_words += '!'
print(stop_words)
for item in root.iterfind("./review/text"):
    print(str(count) + ": " + item.text)
    ArrayList.append(word_tokenize(item.text))
    print(ArrayList[count])
    sentences = sent_tokenize(item.text)
    for unit in sentences:
        word_tokens = word_tokenize(unit)
        List.append([w for w in word_tokens if not w in (stop_words)])

    print(List[count])
    print("\n")
    count += 1

print("\n\n")
print(List)

