import nltk
import xml.etree.ElementTree
import os
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

doc_count = 100
stop_words_file = 'stop_words_tr_147.txt'

def freq_dist(l):
  d = dict()
  for w in l:
    if w in d:
      d[w] += 1
    else:
      d[w] = 1
  return d

def remove_stop_words(stop_words_file_name, words):
  with open(stop_words_file_name, 'r') as myfile:
    stop_words = filter(None, myfile.read().split(' '))
  # convert strings to unicode
  stop_words = [a.decode('unicode-escape') for a in stop_words]
  return [x for x in words if x not in stop_words]

# returns the every word as 5 letter
def f5_stemmer(doc):
  # split doc in to words, remove empty strings
  words = filter(None, doc.split(' '))

  # get first 5 letters for every word
  words = map(lambda x: x[:5], words)
  return words

def get_docs_as_str(doc_num):
  doc_folder = '1k docs'
  doc_names = sorted(os.listdir(doc_folder))
  # print(doc_names)
  r = ""
  for doc in doc_names[:doc_num]:
    e = xml.etree.ElementTree.parse(doc_folder + '/' + doc).getroot()
    doc_str = ""
    for atype in e.findall('TEXT'):
      doc_str += atype.text
    r += doc_str
  return r

# print(get_docs_as_str(2))
doc = get_docs_as_str(1)
word_list = f5_stemmer(doc)
word_list2 = remove_stop_words(stop_words_file, word_list)
print(len(word_list))
print(len(freq_dist(word_list)))

print(len(word_list2))
print(len(freq_dist(word_list2)))

# print (doc_names)
# for atype in e.findall('TEXT'):
#   print(atype.text)