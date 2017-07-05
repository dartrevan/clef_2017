from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy.linalg import norm
import pickle
import codecs

class ArrayTransformer(object):

    def __init__(self): super(ArrayTransformer, self).__init__()

    def fit(self,X, y): pass

    def fit_transform(self, X, y=None): return self.transform(X)

    def transform(self, X): return X.toarray()

def cosine_similarities(raw_texts):
    with codecs.open('nn_models/dictionary_vectors.bin', 'rb') as in_file:
        dictionary_vectors = pickle.load(in_file)
    with codecs.open('nn_models/tfidf_mapper.bin', 'rb') as in_file:
        tfidf_mapper = pickle.load(in_file)

    raw_texts_vectors = tfidf_mapper.transform(raw_texts)

    dictionary_icd_codes_count = dictionary_vectors.shape[0]
    similarities = np.empty((len(raw_texts_vectors), dictionary_icd_codes_count))
    products = raw_texts_vectors.dot(dictionary_vectors.T)
    raw_norms = norm(raw_texts_vectors, axis=1)
    dictionary_norms = norm(dictionary_vectors, axis=1)
    raw_norms[raw_norms == 0] = 1.0
    dictionary_norms[dictionary_norms == 0] = 1.0
    return products/np.expand_dims(raw_norms, axis=1)/np.expand_dims(dictionary_norms, axis=0)

