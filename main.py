import xml.etree.ElementTree
import os
import sys
import numpy as np
import time

from TurkishStemmer import TurkishStemmer 
from collections import defaultdict


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
  with open(stop_words_file_name, 'r', encoding="utf-8") as myfile:
    stop_words = myfile.read().lower().strip().split()

  return [x for x in words if x not in stop_words], set([x for x in words if x in stop_words])

def f5_stemmer(words):
  if len(words) == 0:
    return []
  words = map(lambda x: x[:5], words)
  return words

def other_stemmer(words):
  if len(words) == 0:
    return []
  return other_stemmer_vectorize(words)

@np.vectorize
def other_stemmer_vectorize(words):
  stemmer = TurkishStemmer()
  return stemmer.stem(words)

def get_docs_as_str(doc_num):
  doc_folder = '1k docs'
  doc_names = sorted(os.listdir(doc_folder))

  doc_list = []
  for doc in doc_names[:doc_num]:
    e = xml.etree.ElementTree.parse(doc_folder + '/' + doc).getroot()
    doc_str = ""
    for atype in e.findall('TEXT'):
      doc_str += atype.text
    doc_list.append(doc_str.lower())
  return doc_list

def create_index (data): # https://stackoverflow.com/a/28019844
  index = defaultdict(list)
  for i, tokens in enumerate(data):
      for token in set(tokens):
        index[token].append((i, np.count_nonzero(tokens == token)))

  return index

def main(): 
  for doc_count in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    docs = get_docs_as_str(doc_count)

    list_of_doc_words = []
    stopword_counter = 0
    total_word_count = 0
    set_used_stopwords = set()
    for doc in docs:
      words = doc.strip().split()
      len_before_stopwords = len(words)
      words, set_stopwords = remove_stop_words(stop_words_file, words)

      set_used_stopwords = set_used_stopwords.union(set_stopwords)
      total_word_count += len_before_stopwords
      stopword_counter += len_before_stopwords - len(words)

      words = other_stemmer(words)
      list_of_doc_words.append(words)

    invertedIndexStructure = create_index(list_of_doc_words)
    with open(str(doc_count) + "_turkish_stemmer.txt" , "w") as f:
      termcount = len(invertedIndexStructure)
      f.write("term count: " + str(termcount))
      f.write("\nremoved stopword count: " + str(stopword_counter))
      f.write("\ntotal count: " + str(total_word_count))
      f.write("\nremoved stopwords / total words: " + str(100.0 * stopword_counter / total_word_count))
      f.write("\ninverted index structure size(bytes): " + str(sys.getsizeof(invertedIndexStructure)))
      f.write("\nused stopword percentage: " + str(100.0 * len(set_used_stopwords) / 147))
  return

start_time = time.time()
main()
print(time.time() - start_time)