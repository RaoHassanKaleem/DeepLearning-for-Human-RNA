#!pip install BioPython
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
import re
import csv
import math
import numpy as np
from Bio import SeqIO

def extractFeature(seq):
    max_fatures = 500
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(seq)
    X = tokenizer.texts_to_sequences(seq)
    X = pad_sequences(X)
    return X



