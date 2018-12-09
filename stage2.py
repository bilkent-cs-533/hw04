import numpy as np
import time
import os
import sys
import xml.etree.ElementTree
from TurkishStemmer import TurkishStemmer 
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

stop_words_file = 'stop_words_tr_147.txt'

def remove_stop_words(stop_words_file_name, words):
  with open(stop_words_file_name, 'r', encoding="utf-8") as myfile:
    stop_words = myfile.read().lower().strip().split()

  return [x for x in words if x not in stop_words]

def f5_stemmer(words):
  if len(words) == 0:
    return []
  words = map(lambda x: x[:5], words)
  return words

@np.vectorize
def turkish_stemmer_vectorize(words):
  if len(words) == 0:
    return []
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

def get_docs_as_list(set_size, stemmerId=0):
  docs = get_docs_as_str(set_size)
  
  list_of_doc_words = []
  for doc in docs:
    words = doc.strip().split()
    words = remove_stop_words(stop_words_file, words)

    if stemmerId == 0:
      words = turkish_stemmer_vectorize(words)
    else:
      words = f5_stemmer(words)

    list_of_doc_words.append(words)
  return list_of_doc_words

# docs is list of list of words
def get_corpus(docs):
  d = dict()
  for doc in docs:
    for w in doc:
      d[w] = True
  
  i = 0
  for w in d:
    d[w] = i
    i += 1

  return d

# docs is list of list of words
# corpus is dictionary of word to index
def get_features_as_freq_dist(docs, corpus):
  l = np.zeros((len(docs), len(corpus)))
  for i,doc in enumerate(docs):
    d = dict()
    for word in doc:
      if word in d:
        d[word] += 1
      else:
        d[word] = 1
    for word in doc:
      l[i, corpus[word]] = d[word]
      
  return l

# docs is list of list of words
# corpus is a dictionary of word to document index
def get_features_as_binary_freq_dist(docs, corpus):
  l = np.zeros((len(docs), len(corpus)))
  for i,doc in enumerate(docs):
    for word in doc:
      l[i, corpus[word]] = 1
      
  return l

# avg similarity among members of a cluster
def intra_similarity(clusters, C):
  similarities = dict()
  for i in clusters:
    members = clusters[i]
    n = len(members)
    tot = 0
    for j in range(n):
      for k in range(i+1, n):
        tot += C[j, k]
    similarities[i] = tot/n
  return similarities

def inter_similarity(clusters, C):
  similarities = dict()
  nC = len(clusters)
  for i in range(nC):
    for j in range(i+1, nC):
      similarities[(i,j)] = C[i,j]
  return similarities

def get_D_matrix(set_size, is_bin=True):
  docs = get_docs_as_list(set_size)
  corpus = get_corpus(docs)
  if is_bin:
    return get_features_as_binary_freq_dist(docs, corpus)
  else:
    return get_features_as_freq_dist(docs, corpus)

# returns a dictionary of int to list of int, non-overlapping clustering of documents
def c3m(D):
  
  count_docs = len(D)
  # s1, divide every element with corresponding row sum 
  S1 = D/D.sum(axis=1, keepdims=True)
  # s2, divide every element with corresponding col sum 
  S2 = D/D.sum(axis=0, keepdims=True)

  C = np.dot(S1, S2.transpose())
  nC = int(round(sum(C.diagonal())))
  cluster_seed_powers = np.zeros(count_docs)
  for i in range(count_docs):
    cluster_seed_powers[i] = C[i,i] * (1-C[i,i]) * sum(D[i] != 0)

  sorted_by_seed_power = np.argsort(cluster_seed_powers)
  seed_indices = sorted_by_seed_power[-nC:]
  labels = np.zeros(count_docs)
  for i in seed_indices:
    labels[i] = [i]

  non_seed_indices = sorted_by_seed_power[:count_docs-nC]
  for i in non_seed_indices:
    similarity2seeds = np.take(C[i], seed_indices)
    labels[i] = seed_indices[similarity2seeds.argmax()]

  return labels, C

def affinity_propogation(D):
  af = AffinityPropagation().fit(D)
  cluster_centers_indices = af.cluster_centers_indices_
  labels = af.labels_
  clusters = dict()


def plot_clusters(data, algorithm, args, kwds):
  start_time = time.time()
  labels = algorithm(*args, **kwds).fit_predict(data)
  end_time = time.time()
  palette = sns.color_palette('deep', np.unique(labels).max() + 1)
  colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
  plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
  frame = plt.gca()
  frame.axes.get_xaxis().set_visible(False)
  frame.axes.get_yaxis().set_visible(False)
  plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
  plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
  plt.show()

def mean_shift(D):
  ms = MeanShift()
  ms.fit(D)
  labels = ms.labels_
  cluster_centers = ms.cluster_centers_

  labels_unique = np.unique(labels)
  n_clusters_ = len(labels_unique)

  print("number of estimated clusters : %d" % n_clusters_)

start_time = time.time()
D = get_D_matrix(100, False)

plot_clusters(D, AffinityPropagation, (), {})

print(time.time() - start_time)
start_time = time.time()
# affinity_propogation(D)
print(time.time() - start_time)
