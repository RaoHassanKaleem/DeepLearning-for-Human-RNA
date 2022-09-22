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

def seqToMat(seq):
    max_fatures = 500
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(seq)
    X = tokenizer.texts_to_sequences(seq)
    X = pad_sequences(X)
    return X

def extractFeature (seq):
    mainVec = calcFV(seq.lower())
    encs = replace_seq(seq.lower())
    for x in encs:
        mainVec.append(x)
    fv_array=np.asarray(mainVec).reshape((-1, 822,1))
    return fv_array


