# read in needed libraries
import pandas as pd
import re
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sent2vec
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords')
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
from progressbar import ProgressBar, FormatLabel, Percentage, Bar
import numpy as np


#Load BioSent2vec model
model_path = YOUR_MODEL_LOCATION
model = sent2vec.Sent2vecModel()
try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print('model successfully loaded')

#load Biosent 2 vec word vector (can use vectors of words as look up vs model will make vector if word not in list of vectors)
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format(filename, binary=True)

#load Biosent 2 vec model
from gensim.models import FastText
model = FastText.load_fasttext_format(filename)

#example code to use general (not biosent model) sent2vec model
import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('model.bin')
emb = model.embed_sentence("once upon a time .")
embs = model.embed_sentences(["first sentence .", "another sentence"])


#download and install fastext to use fasttext embeddings
# from this website: https://fasttext.cc/docs/en/supervised-tutorial.html
#   1. created fastText folder in ~/Dropbox/Graduate School/Data and MetaData Integration/fastText/fastText-0.9.1
#   2. downloaded fastText using steps on webpage (weget/cd folder/make fastText etc..)
#   3. Then get the data for fastText from this website and follow directions: https://fasttext.cc/docs/en/unsupervised-tutorial.html
#       a. start with mkdir data section (second box) and make data dir/get data/unzip etc..
#       


def similarity_search(tfidf_matrix, index_var, top_n):
    """
    The function calculates the cosine similarity between the index variables and all other included variables in the
    matrix. The results are sorted and returned as a list of lists, where each list contains a variable identifier
    and the cosine similarity score for the top set of similar variables as indicated by the input argument are
    returned.

    :param tfidf_matrix: where each row represents a variables and each column represents a concept and counts are
    weighted by TF-IDF
    :param index_var: an integer representing a variable id
    :param top_n: an integer representing the number of similar variables to return
    :return: a list of lists where each list contains a variable identifier and the cosine similarity
        score the top set of similar as indicated by the input argument are returned
    """

    # calculate similarity
    cosine_similarities = linear_kernel(tfidf_matrix[index_var:index_var + 1], tfidf_matrix).flatten()
    rel_var_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index_var]
    similar_variables = itertools.islice(((variable, cosine_similarities[variable]) for variable in rel_var_indices), top_n)

    return similar_variables


def preprcessor(data, var_col, defn_col, splitter):
    """
    Using the input two lists, the function assembles an identifier and definition for each question. The definition
    is then preprocessed by making all words lowercase, removing all punctuation, tokenizing by word,
    removing english stop words, and lemmatization (via wordnet). The function returns a list of lists,
    where the first item in each list is the identifier and the second item is a list containing the processed
    question definition.

    :param data: pandas data frame containing variable information
    :param var_col: list of columns used to assemble question identifier
    :param defn_col: list of columns used to assemble question definition
    :param splitter: character to split vector labels. NOTE - if error occurs in output, check this variable and if
    it occurs in the input data change it!
    :return: a list of lists, where the first item in each list is the identifier and the second item is a list
    containing the processed question definition
    """

    widgets = [Percentage(), Bar(), FormatLabel("(elapsed: %(elapsed)s)")]
    pbar = ProgressBar(widgets=widgets, maxval=len(data))

    vocab_dict = []

    for index, row in pbar(data.iterrows()):
        # var = str(row[str(var_col[0])]) + str(splitter) + str(row[str(var_col[1])]) + str(splitter) +\
        #       str(row[str(var_col[2])]) + str(splitter) + str(row[str(var_col[3])])
        var = str(row[str(var_col[0])])


        if len(defn_col) ==1:
            defn = str(row[str(defn_col[0])])
        else:
            defn_col_s = [str(i) for i in row[defn_col]]
            defn = str(" ".join(defn_col_s))

        # lowercase
        defn_lower = defn.lower()

        # tokenize & remove punctuation
        tok_punc_defn = RegexpTokenizer(r"\w+").tokenize(defn_lower)

        # remove stop words & perform lemmatization
        defn_lemma = []
        for x in tok_punc_defn:
            if (all([ord(c) in range(0, 128) for c in x]) and x not in stopwords.words("english")):
                defn_lemma.append(str(WordNetLemmatizer().lemmatize(x)))

        vocab_dict.append((var, defn_lemma))

    pbar.finish()

    # verify all rows were processed before returning data
    if len(vocab_dict) != len(data):
        matched = round(len(vocab_dict) / float(len(data)) * 100, 2)
        raise ValueError('There is a problem - Only matched {0}% of variables were processed'.format(matched))
    else:
        return vocab_dict

