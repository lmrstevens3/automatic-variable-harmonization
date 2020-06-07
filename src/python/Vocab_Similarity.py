##########################################################################################
# Vocab_Similarity.py
# author: TJ Callahan, Laura Stevens
# Purpose: script reads in a csv files of variable documentation including some or all of
#           descriptions, units, as well as coding labels and pairs all variables against all other
#           variables (except against themselves) and scores variable similarity in an attempt to
#          identify which variables, using the documentation, are the most similar.
# version 1.1.1
# python version: 2.7.13
# date: 06.01.2020
##########################################################################################

# read in needed libraries
import itertools

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import WordNetLemmatizer
from progressbar import ProgressBar, FormatLabel, Percentage, Bar
import numpy as np
import scipy.sparse as sp

from sklearn.utils.validation import check_array, FLOAT_DTYPES


nltk.download('stopwords')
nltk.download("wordnet")



def _my_document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)

def my_fit(tfidf, X, y=None):
    """Learn the idf vector (global term weights)

    Parameters
    ----------
    X : sparse matrix, [n_samples, n_features]
        a matrix of term/token counts
    """
    X = check_array(X, accept_sparse=('csr', 'csc'))
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

    if tfidf.use_idf:
        n_samples, n_features = X.shape
        df = _my_document_frequency(X).astype(dtype)

        # perform idf smoothing if required
        df += int(tfidf.smooth_idf)
        n_samples += int(tfidf.smooth_idf)

        # log+1 instead of log makes sure terms with zero idf don't get
        # suppressed entirely.
        idf = np.log(n_samples / df) + 1
        tfidf._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=dtype)

    return tfidf


class MyTfidfVecctorizer(TfidfVectorizer):

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self : TfidfVectorizer
        """
        self._check_params()
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        my_fit(self._tfidf, X)
        return self

class CorpusBuilder:
    def __init__(self, doc_col):

        self.doc_col = doc_col
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.lemmatizer = WordNetLemmatizer()
        self.tf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, use_idf=True, norm="l2",
                                  lowercase=False)
        self.tf = MyTfidfVecctorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, use_idf=True, norm="l2",
                                     lowercase=False)
        self.tfidf_matrix = None
        self.corpus = None
        self.corpora = None

    def lemmatize_variable_documentation(self, var, doc_list):
        # if doc_col is multiple columns,  concatenate text from all columns
        doc = " ".join([str(i) for i in doc_list])

        # tokenize & remove punctuation
        tok_punc_doc = self.tokenizer.tokenize(doc.lower())

        # remove stop words & lemmatize
        doc_lemma = [str(self.lemmatizer.lemmatize(x))
                     for x in tok_punc_doc
                     if all([ord(c) in range(0, 128) for c in x]) and x not in stopwords.words("english")]

        return var, doc_lemma

    def calc_tfidf(self):
        """
        Create a matrix where each row represents a variables and each column represents a word and counts are
        weighted by TF-IDF- matrix is n variable (row) X N all unique words in all documentation (cols)
        :return:
        """
        # BUILD TF-IDF VECTORIZER

        # CREATE MATRIX AND VECTORIZE DATA
        self.fit = self.tf.fit([content for var, content in self.corpus])


        # self.fit.transform([content for corpus in self.corpora for _, content in corpus])

        self.tfidf_matrix = self.tf.fit_transform([content for _, content in self.corpus])

    def build_corpus(self, data, id_col):
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
        var_doc_list = [self.lemmatize_variable_documentation(row[len(self.doc_col)], row[:-1])
                        for row in data[cols].as_matrix()]

        # pbar.finish()

        # verify all rows were processed before returning data
        if len(var_doc_list) != len(data):
            matched = round(len(var_doc_list) / float(len(data)) * 100, 2)
            raise ValueError('There is a problem - Only matched {0}% of variables were processed'.format(matched))
        else:
            self.corpus = var_doc_list


class VariableSimilarityCalculator:

    def __init__(self, data, id_col, filter_data=None, pairable=None, data_cols_to_keep=None,
                 filter_data_cols_to_keep=None, top_n=None,
                 top_n_group=None):
        """

        :param filter_data_cols_to_keep:
        :param filter_data: a pandas data frame containing variable information used to filter results
        containing the processed question definition
        :param data: pandas data frame containing variable information
        :param data_cols_to_keep:
        :param top_n: number of results to return for each variable
        :param id_col: list of columns used to assemble question identifier
        """
        data_col_suffix = "_paired"
        filter_data_col_suffix = "_ref"
        self.filter_data_cols_to_keep = filter_data_cols_to_keep
        self.data_cols_to_keep = data_cols_to_keep

        self.ref_cols = [col + filter_data_col_suffix for col in filter_data_cols_to_keep]
        self.paired_cols = [col + data_col_suffix for col in data_cols_to_keep]
        self.filter_data = filter_data or data
        self.data = data
        self.pairable = pairable or vals_differ_in_col(id_col)
        self.top_n = top_n or len(data.index) - 1  # len(data) - 1  #
        self.top_n_group = top_n_group
        self.id_col = id_col
        self.score_cols = [self.id_col + filter_data_col_suffix,
                           self.id_col.replace(filter_data_col_suffix, "") + data_col_suffix]
        self.cache = None
        self.file_name = None

    def calculate_top_similarity(self):
        return self.top_n

    def similarity_search(self, corpus_builder, ref_var_index):
        """
        The function calculates the cosine similarity between the index variables and all other included variables in
        the matrix. The results are sorted and returned as a list of lists, where each list contains a variable
        identifier and the cosine similarity score for the top set of similar variables as indicated by the input
        argument are returned.

        :param corpus_builder:
        :param ref_var_index: an integer representing a variable id
        :return: a list of lists where each list contains a variable identifier and the cosine similarity
            score the top set of similar as indicated by the input argument are returned
        """

        # calculate similarity
        cosine_similarities = linear_kernel(corpus_builder.tfidf_matrix[ref_var_index:ref_var_index + 1],
                                            corpus_builder.tfidf_matrix).flatten()
        rel_var_indices = [i for i in cosine_similarities.argsort()[::-1] if i != ref_var_index]
        similar_variables = itertools.islice(
            ((i, cosine_similarities[i]) for i in rel_var_indices),
            self.top_n)

        if self.top_n_group:
            pass

        return similar_variables

    def init_cache(self, score_name, file_name=None):
        self.file_name = file_name
        self.score_cols = self.score_cols.append(score_name)
        self.cache = pd.DataFrame([], index=list(self.score_cols).extend(self.data_cols_to_keep))
        self.cache.rename(self.data_cols_to_keep, self.paired_cols)
        self.cache.rename(self.filter_data_cols_to_keep, self.ref_cols)
        if self.file_name:
            with open(self.file_name, "w") as f:
                f.write(",".join(self.cache.index.tolist()))
                f.write("\n")

    def append_cache(self, doc_id_1, d1, doc_id_2, d2, score):
        data = pd.Series([doc_id_1, doc_id_2, score], index=self.score_cols).append(d1).append(d2)

        if self.file_name:
            with open(self.file_name, "a") as f:
                f.write(",".join([str(x) for x in data.values()]))
                f.write("\n")
        elif isinstance(self.cache, pd.DataFrame):
            self.cache.append(data)

    def finalize_cached_output(self):
        if not self.file_name:
            self.cache.to_csv(self.file_name, sep=",", encoding="utf-8", index=False, line_terminator="\n")
        print '\n' + self.file_name + " written"  # " scored size:" + str(len(scored))  # 4013114

    def row_func(self, row_idx, row, corpus_builder, var_idx, matches):
        if var_idx:
            matches += 1
            ref_var_index = var_idx[0]
            doc_id_1 = corpus_builder.corpus[ref_var_index][0]
            d1 = row[self.filter_data_cols_to_keep]
            d1.rename(self.filter_data_cols_to_keep, self.ref_cols)

            # retrieve top_n similar variables
            [self.append_cache(doc_id_1, d1,
                               corpus_builder.corpus[index][0],
                               self.data[self.data_cols_to_keep].iloc([index]).rename(self.data_cols_to_keep,
                                                                                      self.paired_cols),
                               score)
             for index, score in self.similarity_search(corpus_builder, ref_var_index)
             if score > 0 and self.pairable(self.data, index, self.filter_data, row_idx)]

    def score_variables(self, corpus_builder):
        """
        The function iterates over the corpus and returns the top_n (as specified by user) most similar variables,
        with a score, for each variable as a pandas data frame.

        :return: pandas data frame of the top_n (as specified by user) results for each variable
        """

        # cache = init_cache_output(None, "pandas", file_name)

        # matching data in filtered file

        # tqdm.pandas(desc="Filtering the data")
        # filter_data.progress_apply(lambda row: ,axis=1)

        widgets = [Percentage(), Bar(), FormatLabel("(elapsed: %(elapsed)s)")]
        pbar = ProgressBar(widgets=widgets, maxval=len(self.filter_data))
        matches = 0
        for i, row in pbar(self.filter_data.iterrows()):
            var = str(row[str(self.id_col)])
            # get index of filter data in corpus
            var_idx = [x for x, y in enumerate(corpus_builder.corpus) if y[0] == var]
            self.row_func(i, row, corpus_builder, var_idx, matches)

        pbar.finish()

        # verify that we got all the matches we expected (assumes that we should be able to
        # match all vars in filtered data)

        if matches != len(self.filter_data):
            matched = round(matches / float(len(self.filter_data)) * 100, 2)
            raise ValueError('There is a problem - Only matched {0}% of filtered variables'.format(matched))

        self.finalize_cached_output()

        print("Filtering matched " + str(matches) + " of " + str(len(self.filter_data)) + " variables")

    def variable_similarity(self, file_name, score_name, doc_col):
        # PRE-PROCESS DATA & BUILD CORPORA
        # var_col and defn/units/codeLabels_col hold information from the data frame and are used when
        # processing the data

        corpus_builder = CorpusBuilder(doc_col)
        corpus_builder.build_corpus(self.data, self.id_col)
        corpus_builder.calc_tfidf()
        print '\n' + score_name + " tfidf_matrix size:"
        print corpus_builder.tfidf_matrix.shape  # 105611 variables and 33031 unique concepts

        self.init_cache(score_name, file_name)

        # SCORE DATA + WRITE OUT RESULTS
        return self.score_variables(corpus_builder)


def merge_score_results(score_matrix1, score_matrix2, how):
    # determine how many rows should result when merging
    # match = set(list(score_matrix1['matchID'])) - set(list(score_matrix2['matchID']))
    # both = set(list(score_matrix2['matchID'])).intersection(set(list(score_matrix2['matchID'])))

    # merge data - left adding smaller data to larger file
    scored_merged = pd.merge(left=score_matrix1, right=score_matrix2,
                             on=['matchID', 'conceptID', 'study_1', 'dbGaP_dataset_label_1',
                                 'dbGaP_studyID_datasetID_1',
                                 'varID_1', 'var_desc_1', 'timeIntervalDbGaP_1', 'cohort_dbGaP_1', 'metadataID_1',
                                 'study_2', 'dbGaP_dataset_label_2', 'dbGaP_studyID_datasetID_2', 'varID_2',
                                 'var_desc_2', 'timeIntervalDbGaP_2', 'cohort_dbGaP_2', 'metadataID_2'], how=how)

    return scored_merged


def vals_differ_in_col(col):
    return lambda s1, s1_idx, s2, s2_idx: s1[col][s1_idx] != s2[col][s2_idx]


def vals_differ_in_all_cols(cols):
    return lambda s1, s1_idx, s2, s2_idx: all([s1[col][s1_idx] != s2[col][s2_idx] for col in cols])


def val_in_any_row_for_col(col):
    return lambda s1, s1_idx, s2, _: s1[col][s1_idx] in s2[col]


def main():
    dropbox_dir = "/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/"
    metadata_all_vars_file_path = dropbox_dir + "tiff_laura_shared/FHS_CHS_ARIC_MESA_dbGaP_var_report_dict_xml_Info_contVarNA_NLP_timeInterval_noDate_noFU_5-9-19.csv"
    concept_mapped_vars_file_path = dropbox_dir + "CorrectConceptVariablesMapped_contVarNA_NLP.csv"

    # READ IN DATA -- 07.17.19
    data = pd.read_csv(metadata_all_vars_file_path, sep=",", quotechar='"', na_values="",
                       low_memory=False)  # when reading in data, check to see if there is "\r" if
    # not then don't use "lineterminator='\n'", otherwise u
    data.units_1 = data.units_1.fillna("")
    data.dbGaP_dataset_label_1 = data.dbGaP_dataset_label_1.fillna("")
    data.var_desc_1 = data.var_desc_1.fillna("")
    data.var_coding_labels_1 = data.var_coding_labels_1.fillna("")
    len(data)

    # read in filtering file
    filter_data = pd.read_csv(concept_mapped_vars_file_path, sep=",", na_values="", low_memory=False)  # n=700
    filter_data.units_1 = filter_data.units_1.fillna("")
    filter_data.dbGaP_dataset_label_1 = filter_data.dbGaP_dataset_label_1.fillna("")
    filter_data.var_desc_1 = filter_data.var_desc_1.fillna("")
    filter_data.var_coding_labels_1 = filter_data.var_coding_labels_1.fillna("")
    len(filter_data)

    # CODE TO GENERATE RANDOM IDS
    # data["random_id"] = random.sample(range(500000000), len(data))
    # filter_data_m = filter_data.merge(data[['concat', 'random_id']], on='concat', how='inner').reset_index(drop=True)
    # filter_data_m.to_csv("CorrectConceptVariablesMapped_RandomID_12.02.18.csv", sep=",", encoding="utf-8",
    #                      index = False)

    id_col = "varDocID_1"

    save_dir = "tiff_laura_shared/NLP text Score results/"
    # file_name_format = save_dir + "FHS_CHS_MESA_ARIC_text_similarity_scores_%s_ManuallyMappedConceptVars_7.17.19.csv"
    file_name_format = save_dir + "test_%s_vocab_similarity.csv"
    disjoint_col = 'dbGaP_studyID_datasetID_1'
    data_cols_to_keep = ["study_1", 'dbGaP_studyID_datasetID_1', 'dbGaP_dataset_label_1', "varID_1",
                         'var_desc_1', 'timeIntervalDbGaP_1', 'cohort_dbGaP_1']

    filter_data_cols_to_keep = data_cols_to_keep

    my_pred = lambda s1, i1, s2, i2: vals_differ_in_col(disjoint_col)(s1, i1, s2, i2) and val_in_any_row_for_col(disjoint_col)(s1, i1, s2, i2)

    # SCORE DATA + WRITE OUT RESULTS
    calc = VariableSimilarityCalculator(data, id_col,
                                        filter_data=None,  # filter_data
                                        pairable=my_pred,
                                        data_cols_to_keep=data_cols_to_keep,
                                        filter_data_cols_to_keep=filter_data_cols_to_keep)

    score_name = "score_desc"
    file_name = file_name_format % "descOnly"
    doc_col = ["var_desc_1"]
    corpus_builder = CorpusBuilder(doc_col)
    corpus_builder.build_corpus(calc.data, calc.id_col)
    corpus_builder.calc_tfidf()

    print '\n%s tfidf_matrix size %s' % (score_name, str(corpus_builder.tfidf_matrix.shape))

    calc.init_cache(score_name, file_name)

    scored = calc.score_variables(corpus_builder)
    # scored = calc.variable_similarity(file_name, score_name, doc_col)
    len(scored)  # 4013114

    score_name = "score_codeLab"
    file_name = file_name_format % "codingOnly"
    corpus_builder = CorpusBuilder(["var_coding_labels_1"])
    calc.init_cache(score_name, file_name)
    scored_coding = calc.score_variables(corpus_builder)
    # len(scored_coding)

    score_name = "score_units"
    file_name = file_name_format % "unitsOnly_ManuallyMappedConceptVars_7.17.19.csv"
    corpus_builder = CorpusBuilder(["units_1"])
    calc.init_cache(score_name, file_name)
    scored_units = calc.score_variables(corpus_builder)
    # len(scored_units)

    score_name = "score_descUnits"
    file_name = file_name_format % "descUnits_ManuallyMappedConceptVars_7.17.19.csv"
    corpus_builder = CorpusBuilder(["var_desc_1", "units_1"])
    calc.init_cache(score_name, file_name)
    scored_desc_units = calc.score_variables(corpus_builder)
    # len(scored_desc_coding)  # 4013114

    score_name = "score_descCoding"
    file_name = file_name_format % "descCoding_ManuallyMappedConceptVars_7.17.19.csv"
    corpus_builder = CorpusBuilder(["var_desc_1", "var_coding_labels_1"])
    calc.init_cache(score_name, file_name)
    scored_desc_coding = calc.score_variables(corpus_builder)
    # len(scored_desc_coding)  # 4013114

    score_name = "score_descCodingUnits"
    file_name = file_name_format % "descCodingUnits_ManuallyMappedConceptVars_7.17.19.csv"
    corpus_builder = CorpusBuilder(["var_desc_1", "units_1", "var_coding_labels_1"])
    calc.init_cache(score_name, file_name)
    scored_desc_coding_units = calc.score_variables(corpus_builder)
    # len(scored_full) #scored_desc_lab

    # Merge scores files and write to merged file- CURRENTLY "SCORED" data frame is not returned
    # from score_variables-so merged code below will not work with this code.
    # ##############################################################################
    scored_merged = merge_score_results(scored, scored_coding, "outer")
    scored_merged = merge_score_results(scored_merged, scored_units, "outer")
    scored_merged = merge_score_results(scored_merged, scored_desc_units, "outer")
    scored_merged = merge_score_results(scored_merged, scored_desc_coding, "outer")
    scored_merged = merge_score_results(scored_merged, scored_desc_coding_units, "outer")

    scored_merged.to_csv(file_name_format % "All_Scores", sep=",", encoding="utf-8", index=False, line_terminator="\n")


if __name__ == "__main__":
    main()

    varDocFile = "tiff_laura_shared/FHS_CHS_ARIC_MESA_varDoc_dbGaPxmlExtract_timeIntervalAdded_May19_NLPversion.csv"
    manualMappedVarsFile = "data/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv"
    # READ IN DATA -- 07.17.19
    testData = pd.read_csv(varDocFile, sep=",", quotechar='"', na_values="",
                       low_memory=False)  # when reading in data, check
    #  to see if there is "\r" if # not then don't use "lineterminator='\n'", otherwise u
