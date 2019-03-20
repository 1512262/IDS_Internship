import gensim
from gensim import utils
import numpy as np
import sys
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import os

model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz', binary=True)
print('imported model')

def preprocess(text):
    text = str(text)
    text = text.lower()
    doc = text.split("+")
    doc = [word for word in doc if word.isalpha()]
    doc = [word for word in doc if word in model.vocab]
    return doc

def list2vec(l):
    if len(l)==0:
        nanvec = np.empty((300,), float)
        nanvec.fill(np.nan)
        return nanvec
    nl = [model[w] for w in l]
    return np.nanmean(nl,axis=0)

def extract_uniquePhrase(split,df):
    if not os.path.exists("../data/old_VGP/language/"):
        os.makedirs("../data/old_VGP/language/")
    
    fu_VGP = open("../data/old_VGP/language/uniquePhrases_{}_VGP".format(split),"w")
    phrase_VGP=list(set(df.phrase1.tolist()+df.phrase2.tolist()))
    print(split,':',len(phrase_VGP))

    for phrase in phrase_VGP:
        fu_VGP.write(phrase+"\n")

    fu_VGP.close()
    return phrase_VGP

def extract_language_feature(split,df):
    phrase_VGP= extract_uniquePhrase(split,df)

    l2v_VGP = [list2vec(preprocess(phrase)) for phrase in phrase_VGP]
    l2v_VGP = np.array(l2v_VGP)
    print(l2v_VGP.shape)
    np.save("../data/old_VGP/language/WEA_{}_VGP".format(split), l2v_VGP)

def main():
    for split in ['train','val','test']:
        print(split)
        df = pd.read_csv('../data/phrase_pair_{}.csv'.format(split),index_col=0)
        extract_language_feature(split,df)
        print('done')

if __name__ == "__main__":
    main()

