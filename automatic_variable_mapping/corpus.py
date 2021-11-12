from nltk import RegexpTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords
import multiprocessing
from functools import partial
import tqdm
import warnings
from numpy import linalg as la
import numpy as np

from automatic_variable_mapping.tfidf import CorporaTfidfVectorizer

tokenizer = RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()

def lemmatize_variable_documentation(doc_text):  # TODO See if var really needs to be passed here
    # if doc_col is multiple columns,  concatenate text from all columns
    doc = " ".join([str(col) for col in doc_text])

    # tokenize & remove punctuation
    tok_punc_doc = tokenizer.tokenize(doc.lower())

    # remove stop words & lemmatize
    doc_lemma = [str(lemmatizer.lemmatize(x))
                    for x in tok_punc_doc
                    if all([ord(c) in range(0, 128) for c in x]) and x not in stopwords.words("english")]

    return doc_lemma

def calc_tfidf(corpora, vocabulary=None):
    """
    Create a matrix where each row represents a variable and each column represents a word and counts are
    weighted by TF-IDF- matrix is n variable (row) X N all unique words in all documentation (cols)
    :return:
    TFIDFVectorizer with updated matrix of TF-IDF features
    """
    # BUILD TF-IDF VECTORIZER
    vocab = vocabulary or list(set([tok for corpus in corpora for _, doc in corpus for tok in doc]))

    tf = CorporaTfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, use_idf=True, norm="l2", lowercase=False, vocabulary=vocab)

    corpus_all = all_docs(corpora)
    # CREATE MATRIX AND VECTORIZE DATA
    corpora = [[doc for _, doc in corpus] for corpus in corpora]
    tf.fit(corpora)
    return tf.transform(corpus_all)


def calc_doc_embeddings(bow_matrix, vocab, word_vectors):
    """
    :param vocab: a list of unique words in the bow_matrix to extract embeddings for
    :param bow_matrix: a bag of words matrix representing the words in the corpora (full matrix, not sparse matrix)
    :param word_vectors: a genism keyed vectors object containing word embeddings
    :return: a matrix of normalized doc embeddings (n docs X n embedding's dimension)
    """
    # set(vocab).difference(set(word_embeddings.index2word))
    embedding_matrix = np.array([normalize_doc_vectors(word_vectors[word], axis=None)
                                 if word in word_vectors else np.zeros(word_vectors.vector_size) for word in vocab])
    doc_embeddings = normalize_doc_vectors(np.matmul(bow_matrix, embedding_matrix), axis=1)
    return doc_embeddings


def normalize_doc_vectors(doc_vectors, axis):
    norms = la.norm(doc_vectors, axis=axis, keepdims=True)
    if len(np.where(norms == 0)[0]):
        warnings.warn(
            str(np.where(norms == 0)[0].shape[0]) + " documents deleted because document embeddings was all zeros")
        doc_vectors = np.delete(doc_vectors, np.where(norms == 0)[0], axis=0)
        norms = np.delete(norms, np.where(norms == 0)[0], axis=0)
    return doc_vectors / norms


def build_corpora(doc_col, corpora_data, id_col, num_cpus=None):
    """Using a list of dataframes, create a corpus for each dataframe in the list c
    :param id_col: column name of uniqueIDs for documents in the dataframe
    :param corpora_data: a list of dataframes containing the data to be turned into a corpus. Each dataframe in the list
    should have a unique ID that is universal to all dataframes in the list and should have the same doc_cols/id_col names
    :return a list, where each item is the lists of lists returned from build_corpus.
    Build_corpus returns a list of lists where the first item of each list contains the identifier and the
    second item contains the a list containing the processed text for each document/row in the corpus"""

    return [build_corpus(doc_col, corpus_data, id_col, num_cpus=num_cpus) for corpus_data in corpora_data]

def all_docs(corpora):
    return [doc for corpus in corpora for _, doc in corpus]


def helper(doc_col, row):
    return (row[len(doc_col)], lemmatize_variable_documentation(row[:-1]))


def build_corpus(doc_col, corpus_data, id_col, num_cpus=None):
    """
    Using the data and defn_col lists, the function assembles an identifier and text string for each row in data.
    The text string is then preprocessed by making all words lowercase, removing all punctuation,
    tokenizing by word, removing english stop words, and lemmatized (via wordnet).
    The function returns a list of lists,  where the first item in each list is the identifier and
    the second item is a list containing the processed text.

    :return: a list of lists, where the first item in each list is the identifier and the second item is a list
    containing the processed question definition
    """

    # widgets = [Percentage(), Bar(), FormatLabel("(elapsed: %(elapsed)s)")]
    # pbar = ProgressBar(widgets=widgets, maxval=len(data))

    cols = list(doc_col)
    cols.append(id_col)

    if num_cpus:
        pool = multiprocessing.Pool(processes=num_cpus)
        corpus = list (tqdm.tqdm(pool.imap(partial(helper, doc_col), corpus_data[cols].as_matrix()), total=corpus_data.shape[0]))
    else:
        corpus = [(row[len(doc_col)], lemmatize_variable_documentation(row[:-1])) for row in corpus_data[cols].values]

    # verify all rows were processed before returning data
    if len(corpus) != len(corpus_data):
        matched = round(len(corpus) / float(len(corpus_data)) * 100, 2)
        raise ValueError('There is a problem - Only matched {0}% of variables were processed'.format(matched))
    else:
        return corpus
