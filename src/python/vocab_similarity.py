##########################################################################################
# vocab_similarity.py
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
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
# noinspection PyProtectedMember
from sklearn.feature_extraction.text import TfidfVectorizer, _document_frequency
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import WordNetLemmatizer
from progressbar import ProgressBar, FormatLabel, Percentage, Bar
import numpy as np
import scipy.sparse as sp

from sklearn.utils.validation import check_array, FLOAT_DTYPES

nltk.download('stopwords')
nltk.download("wordnet")


def fit_corpora(tfidf, Xs):
    """Learn the idf vector (global term weights)

    Parameters
    ----------
    Xs : list of sparse matricies, X [n_samples, n_features]
        a matrix of term/token counts for each corpus in Xs

    tfidf: TfidfVectorizer._tfidf object
    """
    full_df = 0
    if tfidf.use_idf:
        n_samples, n_features = Xs[0].shape
        dtype = Xs[0].dtype if Xs[0].dtype in FLOAT_DTYPES else np.float64
        for X in Xs:
            X = check_array(X, accept_sparse=('csr', 'csc'))
            if not sp.issparse(X):
                X = sp.csr_matrix(X)

            n_samples, _ = X.shape

            df = _document_frequency(X).astype(dtype)

            # perform idf smoothing if required
            df += int(tfidf.smooth_idf)
            n_samples += int(tfidf.smooth_idf)

            full_df += df / n_samples

        n_corpora = len(Xs)

        # log+1 instead of log makes sure terms with zero idf don't get
        # suppressed entirely.
        idf = np.log(n_corpora / full_df) + 1
        tfidf._idf_diag = sp.diags(idf, offsets=0,
                                   shape=(n_features, n_features),
                                   format='csr',
                                   dtype=dtype)

    return tfidf


class CorporaTfidfVectorizer(TfidfVectorizer):

    def fit(self, corpora, y=None):
        """Learn vocabulary and idf for each corpus in corpora.

        Parameters
        ----------
        corpora : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self : TfidfVectorizer
        :param corpora: corpora list
        :param y:
        """
        self._check_params()
        Xs = [super(TfidfVectorizer, self).fit_transform(raw_documents) for raw_documents in corpora]
        fit_corpora(self._tfidf, Xs)
        return self


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


class VariableSimilarityCalculator:

    def __init__(self, ref_ids, pairable=None, select_scores=lambda scores: scores, score_cols=None):
        """

        :param select_scores:
        :param ref_ids: an iterable containing variable information used to filter results
        containing the processed question definition
        """
        if not score_cols:
            score_cols = ["reference var", "paired var", "score"]
        self.ref_ids = ref_ids
        self.pairable = pairable  # or vals_differ_in_col(id_col)

        self.select_scores = select_scores
        self.score_cols = score_cols
        self.cache = None
        self.file_name = None

    @staticmethod
    def calculate_similarity(tfidf_matrix, ref_var_index):
        """
        The function calculates the cosine similarity between the index variables and all other included variables in
        the matrix. The results are sorted and returned as a list of lists, where each list contains a variable
        identifier and the cosine similarity score for the top set of similar variables as indicated by the input
        argument are returned.

        :param tfidf_matrix:
        :param ref_var_index: an integer representing a variable id
        :return: a list of lists where each list contains a variable identifier and the cosine similarity
            score the top set of similar as indicated by the input argument are returned
        """

        # calculate similarity
        similarities = linear_kernel(tfidf_matrix[ref_var_index:ref_var_index + 1], tfidf_matrix)
        similarities = pd.DataFrame(data=similarities[0], columns=["idx", "score"])
        similarities = similarities.loc[similarities['idx'] != ref_var_index]
        # cosine_similarities = ((i, score) for i, score in enumerate(cosine_similarities) if i != ref_var_index)

        return similarities

    def init_cache(self, file_name=None):
        self.file_name = file_name
        self.cache = pd.DataFrame([], columns=list(self.score_cols))
        if self.file_name:
            with open(self.file_name, "w") as f:
                f.write(",".join(self.score_cols))
                f.write("\n")

    def append_cache(self, ref_doc_id, paired_doc_id, score):
        data = [ref_doc_id, paired_doc_id, score]
        if self.file_name:
            with open(self.file_name, "a") as f:
                f.write(",".join(data))
                f.write("\n")
        else:
            self.cache.append(dict(zip(self.score_cols, data)))

    def finalize_cached_output(self):
        if not self.file_name:
            self.cache.to_csv(self.file_name, sep=",", encoding="utf-8", index=False, line_terminator="\n")
        print '\n' + self.file_name + " written"  # " scored size:" + str(len(scored))  # 4013114

    def cache_sim_scores(self, corpus, ref_id, ref_var_scores):
        # retrieve top_n pairings for reference
        [self.append_cache(ref_id, corpus[i][0], score) for i, score in ref_var_scores]

    def score_variables(self, corpus, tfidf):
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
        pbar = ProgressBar(widgets=widgets, maxval=len(self.ref_ids))
        matches = 0

        corpus_doc_ids = [doc_id for doc_id, _ in corpus]
        for ref_id in pbar(self.ref_ids):
            ref_id = str(ref_id)
            # get index of filter data in corpus
            corpus_ref_idx = corpus_doc_ids.index(ref_id)
            if corpus_ref_idx:
                matches += 1
                ref_var_scores = self.calculate_similarity(tfidf, corpus_ref_idx)
                ref_var_scores = self.select_scores(ref_var_scores)
                ref_var_scores = self.filter_scores(ref_var_scores, ref_id)
                self.cache_sim_scores(corpus, ref_id, ref_var_scores)

        pbar.finish()

        # verify that we got all the matches we expected (assumes that we should be able to
        # match all vars in filtered data)

        if matches != len(self.ref_ids):
            matched = round(matches / float(len(self.ref_ids)) * 100, 2)
            raise ValueError('There is a problem - Only matched {0}% of filtered variables'.format(matched))

        self.finalize_cached_output()

        print("Filtering matched " + str(matches) + " of " + str(len(self.ref_ids)) + " variables")

    def variable_similarity(self, file_name, score_name, doc_col, data, id_col):
        # PRE-PROCESS DATA & BUILD CORPORA
        # var_col and defn/units/codeLabels_col hold information from the data frame and are used when
        # processing the data

        corpus_builder = CorpusBuilder(doc_col)
        corpus_builder.build_corpus(data, id_col)
        corpus_builder.calc_tfidf()
        print '\n' + score_name + " tfidf_matrix size:"
        print corpus_builder.tfidf_matrix.shape  # 105611 variables and 33031 unique concepts

        self.init_cache(file_name)

        # SCORE DATA + WRITE OUT RESULTS
        return self.score_variables(corpus_builder.all_docs(), corpus_builder.tfidf_matrix)

    def filter_scores(self, ref_var_scores, ref_id):
        return ((pair_id, score)
                for pair_id, score in ref_var_scores
                if self.pairable(score, pair_id, self.ref_ids, ref_id))


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


def partition(data, by):
    return [data[data[by] == col_value] for col_value in data[by].unique()]


def vals_differ_in_col(col):
    return lambda s1, s1_idx, s2, s2_idx: s1[col][s1_idx] != s2[col][s2_idx]


def vals_differ_in_all_cols(cols):
    return lambda s1, s1_idx, s2, s2_idx: all([s1[col][s1_idx] != s2[col][s2_idx] for col in cols])


def val_in_any_row_for_col(col):
    return lambda s1, s1_idx, s2, _: s1[col][s1_idx] in s2[col]


def select_top_sims(similarities, n):
    return similarities.sort_values(["score"], ascending=False).take(n)


def select_top_sims_by_group(similarities, n, data, group_col):
    similarities.append(data[group_col])
    return similarities.sort_values(["score"], ascending=False).groupby(by=[group_col]).take(n)


def main():
    dropbox_dir = "/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/"
    metadata_all_vars_file_path = dropbox_dir + "tiff_laura_shared/FCAMD_var_report_NLP_missing_contVars.csv" \
                                                "FHS_CHS_ARIC_MESA_dbGaP_var_report_dict_xml_Info_contVarNA_NLP_timeInterval_noDate_noFU_5-9-19.csv"
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
    # ref_suffix = "_1"
    # pairing_suffix = "_2"
    # score_cols = [id_col + ref_suffix,
    #                 id_col.replace(pairing_suffix, "") + pairing_suffix]
    file_name_format = save_dir + "test_%s_vocab_similarity.csv"
    disjoint_col = 'dbGaP_studyID_datasetID_1'

    # data_cols_to_keep = ["study_1", 'dbGaP_studyID_datasetID_1', 'dbGaP_dataset_label_1', "varID_1",
    #                     'var_desc_1', 'timeIntervalDbGaP_1', 'cohort_dbGaP_1']

    def my_pred(score, s1, i1, s2, i2):
        bools = [f(s1, i1, s2, i2)
                 for f in [val_in_any_row_for_col(disjoint_col),
                           vals_differ_in_col(disjoint_col)]]
        bools.append(score > 0)
        return all(bools)

    top_n = len(data) - 1

    calc = VariableSimilarityCalculator(filter_data[id_col],
                                        pairable=my_pred,
                                        select_scores=lambda sims: select_top_sims_by_group(sims, top_n, data,
                                                                                            disjoint_col))

    score_name = "score_desc"
    calc.score_cols[2] = score_name
    file_name = file_name_format % "descOnly"
    doc_col = ["var_desc_1"]
    corpus_col = "study_1"
    corpus_builder = CorpusBuilder(doc_col)
    corpus_builder.build_corpus(partition(data, by=corpus_col), id_col)
    corpus_builder.calc_tfidf()

    print '\n%s tfidf_matrix size %s' % (score_name, str(corpus_builder.tfidf_matrix.shape))

    calc.init_cache(file_name)

    scored = calc.score_variables(corpus_builder.all_docs(), corpus_builder.tfidf_matrix)
    # scored = calc.variable_similarity(file_name, score_name, doc_col)
    len(scored)  # 4013114

    # score_name = "score_codeLab"
    # file_name = file_name_format % "codingOnly"
    # corpus_builder = CorpusBuilder(["var_coding_labels_1"])
    # calc.init_cache(file_name)
    # scored_coding = calc.score_variables(corpus_builder)
    # # len(scored_coding)
    #
    # score_name = "score_units"
    # file_name = file_name_format % "unitsOnly_ManuallyMappedConceptVars_7.17.19.csv"
    # corpus_builder = CorpusBuilder(["units_1"])
    # calc.init_cache(file_name)
    # scored_units = calc.score_variables(corpus_builder)
    # # len(scored_units)
    #
    # score_name = "score_descUnits"
    # file_name = file_name_format % "descUnits_ManuallyMappedConceptVars_7.17.19.csv"
    # corpus_builder = CorpusBuilder(["var_desc_1", "units_1"])
    # calc.init_cache(file_name)
    # scored_desc_units = calc.score_variables(corpus_builder)
    # # len(scored_desc_coding)  # 4013114
    #
    # score_name = "score_descCoding"
    # file_name = file_name_format % "descCoding_ManuallyMappedConceptVars_7.17.19.csv"
    # corpus_builder = CorpusBuilder(["var_desc_1", "var_coding_labels_1"])
    # calc.init_cache(file_name)
    # scored_desc_coding = calc.score_variables(corpus_builder)
    # # len(scored_desc_coding)  # 4013114
    #
    # score_name = "score_descCodingUnits"
    # file_name = file_name_format % "descCodingUnits_ManuallyMappedConceptVars_7.17.19.csv"
    # corpus_builder = CorpusBuilder(["var_desc_1", "units_1", "var_coding_labels_1"])
    # calc.init_cache(file_name)
    # scored_desc_coding_units = calc.score_variables(corpus_builder)
    # len(scored_full) #scored_desc_lab

    # Merge scores files and write to merged file- CURRENTLY "SCORED" data frame is not returned
    # from score_variables-so merged code below will not work with this code.
    # ##############################################################################
    # scored_merged = merge_score_results(scored, scored_coding, "outer")
    # scored_merged = merge_score_results(scored_merged, scored_units, "outer")
    # scored_merged = merge_score_results(scored_merged, scored_desc_units, "outer")
    # scored_merged = merge_score_results(scored_merged, scored_desc_coding, "outer")
    # scored_merged = merge_score_results(scored_merged, scored_desc_coding_units, "outer")

    # scored_merged.to_csv(file_name_format % "All_Scores", sep=",", encoding="utf-8", index=False, line_terminator="\n")


if __name__ == "__main__":
    main()

    varDocFile = "tiff_laura_shared/FHS_CHS_ARIC_MESA_varDoc_dbGaPxmlExtract_timeIntervalAdded_May19_NLPversion.csv"
    manualMappedVarsFile = "data/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv"
    # READ IN DATA -- 07.17.19
    testData = pd.read_csv(varDocFile, sep=",", quotechar='"', na_values="",
                           low_memory=False)  # when reading in data, check
    #  to see if there is "\r" if # not then don't use "lineterminator='\n'", otherwise u
