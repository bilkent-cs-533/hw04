import numpy as np
import time
import os
import sys
import xml.etree.ElementTree
from TurkishStemmer import TurkishStemmer 
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import scipy

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
    if len(words) == 0:
      continue

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
def intra_sim(labels, S):
  similarities = []
  for c in set(labels):
    totalSim = 0
    totalVal = 0
    for i in range(0,len(S)):
      for j in range(i+1,len(S)):
        if labels[i] == c and labels[j] == c:
          totalSim += S[i][j]
          totalVal += 1
    if totalVal == 0:
      similarities.append(1)
    else: 
      similarities.append(totalSim / totalVal)
  return similarities

def inter_sim(cluster_centers_indices, S):
  similarities = np.zeros((len(cluster_centers_indices),len(cluster_centers_indices)))
  for i, c1 in enumerate(cluster_centers_indices):
    for j, c2 in enumerate(cluster_centers_indices[i+1:]):
      similarities[i][i+j+1] = S[int(c1)][int(c2)]
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
    cluster_seed_powers[i] = C[i,i] * (1-C[i,i]) * sum(D[i])

  sorted_by_seed_power = np.argsort(cluster_seed_powers)
  seed_indices = sorted_by_seed_power[-nC:]
  labels = np.zeros(count_docs)
  for idx,i in enumerate(seed_indices):
    labels[i] = idx

  non_seed_indices = sorted_by_seed_power[:count_docs-nC]
  for i in non_seed_indices:
    similarity2seeds = np.take(C[i], seed_indices)
    labels[i] = similarity2seeds.argmax()

  return labels, seed_indices

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

  labels_unique = np.unique(labels)
  n_clusters_ = len(labels_unique)

  print("number of estimated clusters : %d" % n_clusters_)

def get_inter_intra_sim(D, labels, cluster_center_indices):
  S = 1 - pairwise_distances(D, metric="cosine")
  inter_sim_matrix = inter_sim(cluster_center_indices, S)
  n = inter_sim_matrix.shape[0]
  avg_inter_sim = sum(sum(inter_sim_matrix)) / (n*(n-1)/2)

  intra_sim_matrix = intra_sim(labels, S)
  avg_intra_sim = sum(intra_sim_matrix) / len(intra_sim_matrix)
  print('inter cluster sim: ', avg_inter_sim, ' intra cluster sim: ', avg_intra_sim)
  return avg_inter_sim, avg_intra_sim

start_time0 = time.time()
for set_size in range(100, 600, 100):

  start_time = time.time()
  D = get_D_matrix(set_size, False)
  print('get D matrix for ', set_size, ' executed in ', time.time() - start_time, ' sec')
  
  start_time = time.time()
  af = AffinityPropagation().fit(D)
  labels, cluster_center_indices = af.labels_, af.cluster_centers_indices_
  print('AP for ', set_size, ' executed in ', time.time() - start_time, ' sec')
  
  get_inter_intra_sim(D, labels, cluster_center_indices)
  
  start_time = time.time()
  labels2, cluster_center_indices2 = c3m(D)
  print('c3m for ', set_size, ' executed in ', time.time() - start_time, ' sec')
  
  get_inter_intra_sim(D, labels2, cluster_center_indices2)
  
print('total time ', time.time() - start_time0, ' sec')

