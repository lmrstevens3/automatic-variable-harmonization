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
from typing import List, Any, Tuple

nltk.download('stopwords')
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from progressbar import ProgressBar, FormatLabel, Percentage, Bar


class Calculator:

    def __init__(self, filter_data):
        self.filter_data = filter_data

    def preprocessor(self, data, id_col, defn_col):
        """
        Using the data and defn_col lists, the function assembles an identifier and text string for each row in data.
        The text string is then preprocessed by making all words lowercase, removing all punctuation, tokenizing by word,
        removing english stop words, and lemmatized (via wordnet). The function returns a list of lists,
        where the first item in each list is the identifier and the second item is a list containing the processed text.

        :param data: pandas data frame containing variable information
        :param id_col: list of columns used to assemble question identifier
        :param defn_col: list of columns used to assemble variable definition/documentation
        :return: a list of lists, where the first item in each list is the identifier and the second item is a list
        containing the processed question definition
        """

        widgets = [Percentage(), Bar(), FormatLabel("(elapsed: %(elapsed)s)")]
        pbar = ProgressBar(widgets=widgets, maxval=len(data))

        vocab_dict = []

        for index, row in pbar(data.iterrows()):
            var = str(row[str(id_col[0])])

            # if defn_col is multiple columns,  concatenate text from all columns
            if len(defn_col) == 1:
                defn = str(row[str(defn_col[0])])
            else:
                defn_cols = [str(i) for i in row[defn_col]]  # type: List[str]
                defn = str(" ".join(defn_cols))

            # lowercase
            defn_lower = defn.lower()

            # tokenize & remove punctuation
            tok_punc_defn = RegexpTokenizer(r"\w+").tokenize(defn_lower)

            # remove stop words & lemmatize
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

    def similarity_search(self, tfidf_matrix, ref_var_index, top_n):
        """
        The function calculates the cosine similarity between the index variables and all other included variables in the
        matrix. The results are sorted and returned as a list of lists, where each list contains a variable identifier
        and the cosine similarity score for the top set of similar variables as indicated by the input argument are
        returned.

        :param tfidf_matrix: where each row represents a variables and each column represents a word and counts are
        weighted by TF-IDF- matrix is n variable (row) X N all unique words in all documentation (cols)
        :param ref_var_index: an integer representing a variable id
        :param top_n: an integer representing the number of similar variables to return
        :return: a list of lists where each list contains a variable identifier and the cosine similarity
            score the top set of similar as indicated by the input argument are returned
        """

        # calculate similarity
        cosine_similarities = linear_kernel(tfidf_matrix[ref_var_index:ref_var_index + 1], tfidf_matrix).flatten()
        rel_var_indices = [i for i in cosine_similarities.argsort()[::-1] if i != ref_var_index]
        similar_variables = itertools.islice(
            ((variable, cosine_similarities[variable]) for variable in rel_var_indices),
            top_n)

        return similar_variables

    def init_cache(self, cache, cache_type, file_name):
        if (cache_type == "file"):
            with open(file_name, "w") as f:
                f.write(",".join(cache.index.tolist()))
                f.write("\n")
            return file_name
        else:
            return pd.DataFrame()

    def append_cache(self, cache, my_cols, docID_1, d1, docID_2, d2, score):
        data = pd.Series([docID_1, docID_2, score], index=my_cols).append(d1).append(d2)
        if (isinstance(cache, str)):
            with  open(cache, "a") as f:
                f.write(",".join([str(x) for x in data.values()]))
                f.write("\n")
            return cache
        elif (isinstance(cache, pd.DataFrame)):
            return cache.append(data)

    def finalize_cached_output(file_name, cache):
        if (isinstance(cache, str)):
            return cache
        else:
            cache.to_csv(file_name, sep=",", encoding="utf-8", index=False, line_terminator="\n")
            return cache

    def row_func(self, row, data, corpus, tfidf_matrix, top_n, var_idx, disjoint_col, data_cols, my_cols, cache,
                 matches):
        if var_idx:
            matches += 1
            ref_var_index = var_idx[0]
            docID_1 = corpus[ref_var_index][0]
            d1 = row[data_cols]

            # retrieve top_n similar variables
            [self.append_cache(cache, my_cols, docID_1, d1, corpus[index][0], data[data_cols].iloc([index]), score)
             for index, score in self.similarity_search(tfidf_matrix, ref_var_index, top_n)
             if score > 0 and data[disjoint_col][index] != row[disjoint_col]]

    def score_variables(self, score_name, data, corpus, tfidf_matrix, top_n, file_name,
                        id_col, disjoint_col, data_cols):
        """
        The function iterates over the corpus and returns the top_n (as specified by user) most similar variables,
        with a score, for each variable as a pandas data frame.

        :param id_col: list of columns used to assemble question identifier
        :param data: pandas data frame containing variable information
        :param corpus: a list of lists, where the first item in each list is the identifier and the second item is a list
        containing the processed question definition
        :param filter_data: a pandas data frame containing variable information used to filter results
        containing the processed question definition
        :param tfidf_matrix: matrix where each row represents a variables and each column represents a concept and counts
        are weighted by TF-IDF
        :param top_n: number of results to return for each variable
        :return: pandas data frame of the top_n (as specified by user) results for each variable
        """

        my_cols = [id_col[0], id_col[0].replace("_1", "_2"), score_name]

        cache = self.init_cache(pd.Series([], index=list(my_cols).extend(data_cols)), "file", file_name)
        # cache = init_cache_output(None, "pandas", file_name)

        # matching data in filtered file

        # tqdm.pandas(desc="Filtering the data")
        # filter_data.progress_apply(lambda row: ,axis=1)

        widgets = [Percentage(), Bar(), FormatLabel("(elapsed: %(elapsed)s)")]
        pbar = ProgressBar(widgets=widgets, maxval=len(self.filter_data))
        matches = 0
        for _, row in pbar(self.filter_data.iterrows()):
            var = str(row[str(id_col[0])])
            # get index of filter data in corpus
            var_idx = [x for x, y in enumerate(corpus) if y[0] == var]
            self.row_func(row, data, corpus, tfidf_matrix, top_n, var_idx, disjoint_col, data_cols, my_cols, cache,
                          matches)

        pbar.finish()

        # verify that we got all the matches we expected (assumes that we should be able to match all vars in filtered data)

        if matches != len(self.filter_data):
            matched = round(matches / float(len(self.filter_data)) * 100, 2)
            raise ValueError('There is a problem - Only matched {0}% of filtered variables'.format(matched))

        self.finalize_cached_output(file_name, cache)

        print("Filtering matched " + str(matches) + " of " + str(len(self.filter_data)) + " variables")
        return cache

    def variable_similarity(self, data, var_col, dataColsList, score_name, fileOut):
        ## PRE-PROCESS DATA & BUILD CORPORA
        # var_col and defn/units/codeLabels_col hold information from the data frame and are used when processing the data
        corpus = self.preprocessor(data, var_col, dataColsList)

        ## BUILD TF-IDF VECTORIZER
        tf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, use_idf=True, norm="l2", lowercase=False)

        ## CREATE MATRIX AND VECTORIZE DATA
        tfidf_matrix = tf.fit_transform([content for var, content in corpus])
        print '\n' + score_name + " tfidf_matrix size:"
        print tfidf_matrix.shape  # 105611 variables and 33031 unique concepts

        ## SCORE DATA + WRITE OUT RESULTS
        disjoint_col = 'dbGaP_studyID_datasetID_1'
        data_cols = ["study_1", 'dbGaP_studyID_datasetID_1', 'dbGaP_dataset_label_1', "varID_1",
                     'var_desc_1', 'timeIntervalDbGaP_1', 'cohort_dbGaP_1']
        scored = self.score_variables(score_name, data, corpus, tfidf_matrix, len(data) - 1, fileOut,
                                      var_col, disjoint_col, data_cols)
        print '\n' + fileOut + " written"  # " scored size:" + str(len(scored))  # 4013114
        return (scored)


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

    return (scored_merged)


def main():
    metadataAllVarsFilePath = "/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/tiff_laura_shared/FHS_CHS_ARIC_MESA_dbGaP_var_report_dict_xml_Info_contVarNA_NLP_timeInterval_noDate_noFU_5-9-19.csv"
    conceptMappedVarsFilePath = "/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/CorrectConceptVariablesMapped_contVarNA_NLP.csv"
    ## READ IN DATA -- 07.17.19
    data = pd.read_csv(metadataAllVarsFilePath, sep=",", quotechar='"', na_values="",
                       low_memory=False)  # when reading in data, check to see if there is "\r" if
    # not then don't use "lineterminator='\n'", otherwise u
    data.units_1 = data.units_1.fillna("")
    data.dbGaP_dataset_label_1 = data.dbGaP_dataset_label_1.fillna("")
    data.var_desc_1 = data.var_desc_1.fillna("")
    data.var_coding_labels_1 = data.var_coding_labels_1.fillna("")
    len(data)

    # read in filtering file
    filter_data = pd.read_csv(conceptMappedVarsFilePath, sep=",", na_values="", low_memory=False)  # n=700
    filter_data.units_1 = filter_data.units_1.fillna("")
    filter_data.dbGaP_dataset_label_1 = filter_data.dbGaP_dataset_label_1.fillna("")
    filter_data.var_desc_1 = filter_data.var_desc_1.fillna("")
    filter_data.var_coding_labels_1 = filter_data.var_coding_labels_1.fillna("")
    len(filter_data)

    ## CODE TO GENERATE RANDOM IDS
    # data["random_id"] = random.sample(range(500000000), len(data))
    # filter_data_m = filter_data.merge(data[['concat', 'random_id']], on='concat', how='inner').reset_index(drop=True)
    # filter_data_m.to_csv("CorrectConceptVariablesMapped_RandomID_12.02.18.csv", sep=",", encoding="utf-8",
    #                      index = False)

    var_col = ["varDocID_1"]
    sep = ","

    "FHS_CHS_MESA_ARIC_text_similarity_scores"

    descFileOut = "tiff_laura_shared/NLP text Score results/FHS_CHS_MESA_ARIC_text_similarity_scores_descOnly_ManuallyMappedConceptVars_7.17.19.csv"
    codingFileOut = "tiff_laura_shared/NLP text Score results/FHS_CHS_MESA_ARIC_text_similarity_scores_codingOnly_ManuallyMappedConceptVars_7.17.19.csv"
    unitsFileOut = "tiff_laura_shared/NLP text Score results/FHS_CHS_MESA_ARIC_text_similarity_scores_unitsOnly_ManuallyMappedConceptVars_7.17.19.csv"
    desc_unitsFileOut = "tiff_laura_shared/NLP text Score results/FHS_CHS_MESA_ARIC_text_similarity_scores_descUnits_ManuallyMappedConceptVars_7.17.19.csv"
    desc_codingFileOut = "tiff_laura_shared/NLP text Score results/FHS_CHS_MESA_ARIC_text_similarity_scores_descCoding_ManuallyMappedConceptVars_7.17.19.csv"
    allFileOut = "tiff_laura_shared/NLP text Score results/FHS_CHS_MESA_ARIC_text_similarity_scores_descCodingUnits_ManuallyMappedConceptVars_7.17.19.csv"

    ## SCORE DATA + WRITE OUT RESULTS
    calc = Calculator(filter_data)
    scored = calc.variable_similarity(data, var_col, ["var_desc_1"], sep, "score_desc", descFileOut)
    # len(scored) #4013114

    scored_coding = calc.variable_similarity(data, var_col, ["var_coding_labels_1"], sep, "score_codeLab",
                                             codingFileOut)
    # len(scored_coding)

    scored_units = calc.variable_similarity(data, var_col, ["units_1"], sep, "score_units", unitsFileOut)
    # len(scored_units)

    scored_desc_units = calc.variable_similarity(data, var_col, ["var_desc_1", "units_1"], sep,
                                                 "score_descUnits", desc_unitsFileOut)
    # len(scored_desc_coding)  # 4013114

    scored_desc_coding = calc.variable_similarity(data, var_col, ["var_desc_1", "var_coding_labels_1"], sep,
                                                  "score_descCoding", desc_codingFileOut)
    # len(scored_desc_coding)  # 4013114

    scored_desc_coding_units = calc.variable_similarity(data, var_col, ["var_desc_1", "units_1", "var_coding_labels_1"],
                                                        sep, "score_descCodingUnits", allFileOut)
    # len(scored_full) #scored_desc_lab

    # Merge scores files and write to merged file- CURRENTLY "SCORED" data frame is not returned from score_variables-so merged code below will not work with this code.
    # ##############################################################################
    scored_merged = merge_score_results(scored, scored_coding, "outer")
    scored_merged = merge_score_results(scored_merged, scored_units, "outer")
    scored_merged = merge_score_results(scored_merged, scored_desc_units, "outer")
    scored_merged = merge_score_results(scored_merged, scored_desc_coding, "outer")
    scored_merged = merge_score_results(scored_merged, scored_desc_coding_units, "outer")

    scored_merged.to_csv("tiff_laura_shared/NLP text Score "
                         "results/FHS_CHS_MESA_ARIC_text_similarity_scores_All_Scores_MannuallyMappedConceptVars_7.17.19.csv",
                         sep=",",
                         encoding="utf-8", index=False, line_terminator="\n")


if __name__ == "__main__":
    main()

varDocFile = "tiff_laura_shared/FHS_CHS_ARIC_MESA_varDoc_dbGaPxmlExtract_timeIntervalAdded_May19_NLPversion.csv"
manualMappedVarsFile = "data/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv"
## READ IN DATA -- 07.17.19
testData = pd.read_csv(varDocFile, sep=",", quotechar='"', na_values="",
                       low_memory=False)  # when reading in data, check
#  to see if there is "\r" if # not then don't use "lineterminator='\n'", otherwise u
