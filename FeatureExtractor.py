#!pip install BioPython
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import csv
import math
import numpy as np
from Bio import SeqIO

def seqToMat(seq):
    max_fatures = 500
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(seq.values)
    X = tokenizer.texts_to_sequences(seq.values)
    X = pad_sequences(X)
    return X