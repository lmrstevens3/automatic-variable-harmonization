from nltk import RegexpTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords

from tfidf import CorporaTfidfVectorizer


class CorpusBuilder:
    def __init__(self, doc_col):

        self.doc_col = doc_col
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.lemmatizer = WordNetLemmatizer()
        # self.tfidf_type =   None#tfidf_type or "single_corpus"
        # self.representation = representation or "tfidf"
        self.tf = None
        self.tfidf_matrix = None
        self.corpora = None

    def lemmatize_variable_documentation(self, var, doc_text):
        # if doc_col is multiple columns,  concatenate text from all columns
        doc = " ".join([str(i) for i in doc_text])

        # tokenize & remove punctuation
        tok_punc_doc = self.tokenizer.tokenize(doc.lower())

        # remove stop words & lemmatize
        doc_lemma = [str(self.lemmatizer.lemmatize(x))
                     for x in tok_punc_doc
                     if all([ord(c) in range(0, 128) for c in x]) and x not in stopwords.words("english")]

        return var, doc_lemma

    def calc_tfidf(self, vocabulary=None):
        """
        Create a matrix where each row represents a variables and each column represents a word and counts are
        weighted by TF-IDF- matrix is n variable (row) X N all unique words in all documentation (cols)
        :return:
        TFIDFVectorizer with updated matrix of TF-IDF features
        """
        # BUILD TF-IDF VECTORIZER
        vocab = vocabulary or list(set([tok for corpus in self.corpora for _, doc in corpus for tok in doc]))

        self.tf = CorporaTfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, use_idf=True, norm="l2",
                                         lowercase=False, vocabulary=vocab)

        corpus_all = self.all_docs()
        # CREATE MATRIX AND VECTORIZE DATA
        corpora = [[doc for _, doc in corpus] for corpus in self.corpora]
        self.tf.fit(corpora)
        self.tfidf_matrix = self.tf.transform(corpus_all)

    def build_corpora(self, corpora_data, id_col):
        """Using a list of dataframes, create a corpus for each dataframe in the list c
        :param id_col: column name of uniqueIDs for documents in the dataframe
        :param corpora_data: a list of dataframes containing the data to be turned into a corpus. Each dataframe in the list
        should have a unique ID that is universal to all dataframes in the list and should have the same doc_cols/id_col names
        :return a list, where each item is the lists of lists returned from build_corpus.
        Build_corpus returns a list of lists where the first item of each list contains the identifier and the
        second item contains the a list containing the processed text for each document/row in the corpus"""

        corpora = [self.build_corpus(corpus_data, id_col) for corpus_data in corpora_data]
        self.corpora = corpora

    def all_docs(self):
        return [doc for corpus in self.corpora for _, doc in corpus]

    def build_corpus(self, corpus_data, id_col):
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

        cols = list(self.doc_col)
        cols.append(id_col)
        corpus = [self.lemmatize_variable_documentation(row[len(self.doc_col)], row[:-1])
                  for row in corpus_data[cols].as_matrix()]

        # pbar.finish()

        # verify all rows were processed before returning data
        if len(corpus) != len(corpus_data):
            matched = round(len(corpus) / float(len(corpus_data)) * 100, 2)
            raise ValueError('There is a problem - Only matched {0}% of variables were processed'.format(matched))
        else:
            return corpus
