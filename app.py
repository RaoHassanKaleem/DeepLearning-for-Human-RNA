from io import StringIO
from Bio import SeqIO
import pandas as pd
import streamlit as st
from PIL import Image
import FeatureExtractor as fe
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import numpy as np

icon = Image.open('fav.png')
st.set_page_config(page_title='LSTM-Deep', page_icon = icon)

def seqValidator(seq):
    for i in range(len(seq)):
        if (seq[i] != 'A' and seq[i] != 'G' and seq[i] != 'C' and seq[i] != 'T' and seq[i] != 'a' and seq[i] != 'g' and seq[i] != 'c' and seq[i] != 't'):
            return False
    return True

def createModel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.load_weights("model.h5")
    return model

final_df = pd.DataFrame(columns=['Sequence ID', 'Sequence','Indices','Label'])
seq = ""
len_seq = 0
image = Image.open('WebPic.jpg')
st.subheader("""LSTM-Deep""")
st.image(image, use_column_width=True)
st.sidebar.subheader(("Input Sequence(s) of Length 41 or greater (FASTA FORMAT ONLY)"))
fasta_string  = st.sidebar.text_area("Sequence Input", height=200)
            
st.subheader("Click the Example Button for Sample Data")

if st.button('Example'):
    st.info("Positive Sequences")
    st.code(">Seq1\UGCCGAGACCUCAAGGCCAAGGGAAUUUUAUUUGUGGGGAG", language="markdown")
    st.code(">Seq2\GGCCUGAAAAACAUGAUCAAGGAGGUGGAUGAGGACUUUGA", language="markdown")
    st.info("Negative Sequences")
    st.code(">Seq1\AAACUCCUUCAUCCAAGUCUGGUUCUUCUCCUCUUGGUCCU", language="markdown")
    st.code(">Seq2\AAAUCAGGUGGAAGGACAGAGCACCCUUUCACCGUGGAGGA", language="markdown")


if st.sidebar.button("SUBMIT"):
    if(fasta_string==""):
        st.info("Please input the sequence first. Length should be 41 or greater.")
    fasta_io = StringIO(fasta_string) 
    records = SeqIO.parse(fasta_io, "fasta") 
    for rec in records:
        seq_id = str(rec.id)
        seq=str(rec.seq)
        if(seqValidator(seq)):
            len_seq = len(seq)
            if (len_seq < 41):
                st.info("Please input the sequence again. Length should be 41 or greater. Currently length is " + str(len_seq))
            elif (len_seq == 41):
                df_temp = pd.DataFrame([[seq_id, seq,'Complete(1-41)','None']], columns=['Sequence ID', 'Sequence','Indices','Label'] )
                final_df = pd.concat([final_df,df_temp], ignore_index=True)
            else:
                n_seqs = len_seq - 41
                for i in range(n_seqs + 1):
                    sub_seq = seq[i: i+41]
                    df_temp = pd.DataFrame([[seq_id, sub_seq,str(i+1)+'-'+str(i+300),'None']], columns=['Sequence ID', 'Sequence','Indices','Label'] )
                    final_df = pd.concat([final_df,df_temp], ignore_index=True)
        else:
            st.info("Sequence with Sequence ID: " + str(seq_id) + " is invalid, containing letters other than A,G,C,T.")
    fasta_io.close()
    if(final_df.shape[0]!=0):
        model = createModel()
        for iter in range(final_df.shape[0]):
            temp_seq =  final_df.iloc[iter, 1]
            fv_array = fe.extractFeature(temp_seq)
            score = model.predict(fv_array)
            pred_label = np.round_(score, decimals=0, out=None)
            if(pred_label==1):
                pred_label="Positive"
            else:
                pred_label="Negative"
            final_df.iloc[iter, 3] = str(pred_label)

    st.dataframe(final_df)