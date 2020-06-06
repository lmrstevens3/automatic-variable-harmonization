##########################################################################################
# Vocab_Similarity.py
# author: TJ Callahan
# Purpose: script reads in a csv files of variable labels and names and aims to
#          identify which variables, using the string, are the most similar.
# version 1.1.0
# python version: 2.7.13
# date: 09.27.2018
##########################################################################################


# read in needed libraries
import pandas as pd
import re
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords')
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from progressbar import ProgressBar, FormatLabel, Percentage, Bar
import numpy as np


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


def preprocessor(data, var_col, defn_col, splitter):
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


def score_variables(score_name, var_col, data, filter_data, corpus, tfidf_matrix, top_n, fileOut):
    """
    The function iterates over the corpus and returns the top_n (as specified by user) most similar variables,
    with a score, for each variable as a pandas data frame.

    :param var_col: list of columns used to assemble question identifier
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

    widgets = [Percentage(), Bar(), FormatLabel("(elapsed: %(elapsed)s)")]
    pbar = ProgressBar(widgets=widgets, maxval=len(filter_data))
    with open(fileOut,'w') as sim_res:
        matches = 0
        sim_res.write(",".join([
             "metadataID_1",
             "study_1",
             "studyID_datasetID_1",
             "dbGaP_dataset_label_1",
             "varID_1",
             "var_desc_1",
             "timeIntervalDbGaP_1",
             "cohort_dbGaP_1",
             "metadataID_2",
             "study_2",
             "studyID_datasetID_2",
             "dbGaP_dataset_label_2",
             "varID_2",
             "var_desc_2",
             "timeIntervalDbGaP_2",
             "cohort_dbGaP_2",
             score_name,
             "matchID", "\n"]))

        # matching data in filtered file
        for num, row in pbar(filter_data.iterrows()):
            # var = str(row[str(var_col[0])]) + str(splitter) + str(row[str(var_col[1])]) + str(splitter) + \
            #       str(row[str(var_col[2])]) + str(splitter) + str(row[str(var_col[3])])
            var = str(row[str(var_col[0])])

            # get index of filter data in corpus
            var_idx = [x for x, y in enumerate(corpus) if y[0] == var]

            if var_idx:
                matches += 1
                randID = corpus[var_idx[0]][0]
                study = row['study_1']
                varID = row['varID_1']
                studyID_datasetID = row['studyID_datasetID_1']
                dbGaP_dataset_label = row['dbGaP_dataset_label_1']
                var_desc = row["var_desc_1"]
                timeIntervalDbGaP = row['timeIntervalDbGaP_1']
                cohort_dbGaP = row['cohort_dbGaP_1']
                #conceptID = row["conceptID"]

                # retrieve top_n similar variables
                for index, score in similarity_search(tfidf_matrix, var_idx[0], top_n):
                    if score > 0:
                        randID_2 = corpus[index][0]
                        study_2 = data["study_1"][index]
                        varID_2 = data["varID_1"][index]
                        studyID_datasetID_2 = data['studyID_datasetID_1'][index]
                        dbGaP_dataset_label_2 = data['dbGaP_dataset_label_1'][index]
                        timeIntervalDbGaP_2 = data['timeIntervalDbGaP_1'][index]
                        cohort_dbGaP_2 = data['cohort_dbGaP_1'][index]
                        var_desc_2 = data["var_desc_1"][index]
                        matchID = str(randID) + "_" + str(randID_2)
                        if(studyID_datasetID_2 != studyID_datasetID):
                        #conceptID,
                            sim_res.append(
                                [randID, study, studyID_datasetID, dbGaP_dataset_label, varID, var_desc,
                                 timeIntervalDbGaP, cohort_dbGaP,
                                 randID_2, study_2, studyID_datasetID_2, dbGaP_dataset_label_2, varID_2, var_desc_2,
                                 timeIntervalDbGaP_2, cohort_dbGaP_2,
                                 score, matchID])
                            #sim_res.write(",".join([str(x) for x in [randID,  study, studyID_datasetID, dbGaP_dataset_label, varID, str(var_desc), timeIntervalDbGaP, cohort_dbGaP,
                            #            randID_2, study_2, studyID_datasetID_2, dbGaP_dataset_label_2, varID_2, str(var_desc_2), timeIntervalDbGaP_2, cohort_dbGaP_2,
                            #            score, matchID]]))
                            #sim_res.write("\n")
                #create want to group_by and filter by dataset to get top 10 matches
                #####MAKE THIS A FUNCTION SO YOU CAN SET THE CUT OFF AND SET THE GROUPING VAR
                # create pandas dataframe
                this_varID_matches = pd.DataFrame(dict(metadataID_1=[x[0] for x in sim_res],
                                                 #conceptID=[x[1] for x in sim_res],
                                                 study_1=[x[2] for x in sim_res],
                                                 studyID_datasetID_1=[x[3] for x in sim_res],
                                                 dbGaP_dataset_label_1=[x[4] for x in sim_res],
                                                 varID_1=[x[5] for x in sim_res],
                                                 var_desc_1=[x[6] for x in sim_res],
                                                 timeIntervalDbGaP_1=[x[7] for x in sim_res],
                                                 cohort_dbGaP_1=[x[8] for x in sim_res],
                                                 metadataID_2=[x[9] for x in sim_res],
                                                 study_2=[x[10] for x in sim_res],
                                                 studyID_datasetID_2=[x[11] for x in sim_res],
                                                 dbGaP_dataset_label_2=[x[12] for x in sim_res],
                                                 varID_2=[x[13] for x in sim_res],
                                                 var_desc_2=[x[14] for x in sim_res],
                                                 timeIntervalDbGaP_2=[x[15] for x in sim_res],
                                                 cohort_dbGaP_2=[x[16] for x in sim_res],
                                                 score_name=[x[17] for x in sim_res],
                                                 matchID=[x[18] for x in sim_res]))
                top_matches_perGroup = this_varID_matches.sort_values([score_name], ascending=False).groupby(["studyID_datasetID_2"], sort=False)
                top_matches_perGroup_ranked_this_varID = top_matches_perGroup.rank(method="dense").head(top_n)
            if scored_vars:
                scored_vars = pd.concat([scored_vars, top_matches_perGroup_ranked_this_varID])
            else:
                scored_vars = top_matches_perGroup_ranked_this_varID

    # verify that we got all the matches we expected (assumes that we should be able to match all vars in filtered data)
    if matches != len(filter_data):
        matched = round(matches/float(len(filter_data))*100, 2)
        raise ValueError('There is a problem - Only matched {0}% of filtered variables'.format(matched))
    else:
        print("Filtering matched " + str(matches) + " of " + str(len(filter_data)) + " variables")
        scored_vars.to_csv(fileOut, sep=",",
                           encoding="utf-8", index=False, line_terminator="\n")
        return scored_vars


def variable_similarity(data, var_col, dataColsList, filter_data, sep, score_name, fileOut):
    ## PRE-PROCESS DATA & BUILD CORPORA
    # var_col and defn/units/codeLabels_col hold information from the data frame and are used when processing the data
    corpus = preprocessor(data, var_col, dataColsList, sep)

    ## BUILD TF-IDF VECTORIZER
    tf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, use_idf=True, norm="l2", lowercase=False)

    ## CREATE MATRIX AND VECTORIZE DATA
    tfidf_matrix = tf.fit_transform([content for var, content in corpus])
    print '\n' + score_name + " tfidf_matrix size:"
    print tfidf_matrix.shape  # 105611 variables and 33031 unique concepts

    ## SCORE DATA + WRITE OUT RESULTS
    scored = score_variables(score_name, var_col, data, filter_data, corpus, tfidf_matrix, len(data) - 1, fileOut)
    print '\n' + fileOut + " written"#" scored size:" + str(len(scored))  # 4013114
    return(scored)


def merge_score_results(score_matrix1, score_matrix2, how):
    # determine how many rows should result when merging
    #match = set(list(score_matrix1['matchID'])) - set(list(score_matrix2['matchID']))
    #both = set(list(score_matrix2['matchID'])).intersection(set(list(score_matrix2['matchID'])))

    # merge data - left adding smaller data to larger file 'conceptID',
    scored_merged = pd.merge(left=score_matrix1, right=score_matrix2,
                             on=['matchID',  'study_1', 'dbGaP_dataset_label_1','studyID_datasetID_1',
                                 'varID_1','var_desc_1', 'timeIntervalDbGaP_1','cohort_dbGaP_1','metadataID_1',
                                 'study_2', 'dbGaP_dataset_label_2','studyID_datasetID_2', 'varID_2',
                                 'var_desc_2', 'timeIntervalDbGaP_2','cohort_dbGaP_2','metadataID_2'], how=how)

    return(scored_merged)

def main():

    metadataAllVarsFilePath = "/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/tiff_laura_shared/FHS_CHS_ARIC_MESA_dbGaP_var_report_dict_xml_Info_contVarNA_NLP_timeInterval_noDate_noFU_5-9-19.csv"
    conceptMappedVarsFilePath = "/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/CorrectConceptVariablesMapped_contVarNA_NLP.csv"
    ## READ IN DATA -- 09.27.18
    data = pd.read_csv(metadataAllVarsFilePath , sep=",", quotechar='"', na_values="", low_memory=False) #when reading in data, check to see if there is "\r" if
    # not then don't use "lineterminator='\n'", otherwise u
    data.units_1 = data.units_1.fillna("")
    data.dbGaP_dataset_label_1 = data.dbGaP_dataset_label_1.fillna("")
    data.var_desc_1 = data.var_desc_1.fillna("")
    data.var_coding_labels_1 = data.var_coding_labels_1.fillna("")
    len(data)

    # read in filtering file
    #filter_data = pd.read_csv(conceptMappedVarsFilePath, sep=",", na_values="", low_memory=False) #n=700
    #filter_data.units_1 = filter_data.units_1.fillna("")
    #filter_data.dbGaP_dataset_label_1 = filter_data.dbGaP_dataset_label_1.fillna("")
    #filter_data.var_desc_1 = filter_data.var_desc_1.fillna("")
    #filter_data.var_coding_labels_1 = filter_data.var_coding_labels_1.fillna("")
    #len(filter_data)
    filter_data = data
    ## CODE TO GENERATE RANDOM IDS
    # data["random_id"] = random.sample(range(500000000), len(data))
    # filter_data_m = filter_data.merge(data[['concat', 'random_id']], on='concat', how='inner').reset_index(drop=True)
    # filter_data_m.to_csv("CorrectConceptVariablesMapped_RandomID_12.02.18.csv", sep=",", encoding="utf-8",
    #                      index = False)

    var_col = ["metadataID_1"]
    sep = "`"

    descFileOut = "tiff_laura_shared/NLP text Score results/All_Study_Similar_Variables_filtered_allMatches_varDescOnly_7.17.19.csv"
    codelabFileOut = "tiff_laura_shared/NLP text Score results/All_Study_Similar_Variables_allMatches_codeLabelsOnly_7.17.19.csv"
    unitsFileOut = "tiff_laura_shared/NLP text Score results/All_Study_Similar_Variables_unitsOnly_7.17.19.csv"
    desc_unitsFileOut = "tiff_laura_shared/NLP text Score results/All_Study_Similar_Variables_filtered_allMatches_varDesc_units_7.17.19.csv"
    desc_codelabFileOut = "tiff_laura_shared/NLP text Score results/All_Study_Similar_Variables_allMatches_varDescCodeLabels_7.17.19.csv"
    allFileOut = "tiff_laura_shared/NLP text Score results/All_Study_Similar_Variables_allMatches_all_7.17.19.csv"

    ## SCORE DATA + WRITE OUT RESULTS
    scored = variable_similarity(data, var_col, ["var_desc_1"], filter_data, sep, "score_desc", descFileOut)
    # len(scored) #4013114

    scored_codelab = variable_similarity(data, var_col, ["var_coding_labels_1"], filter_data, sep, "score_codeLab",
                                         codelabFileOut)
    # len(scored_codelab)

    scored_units = variable_similarity(data, var_col, ["units_1"], filter_data, sep, "score_unitsOnly",
                                       unitsFileOut)
    # len(scored_units)


    scored_desc_units = variable_similarity(data, var_col, ["var_desc_1", "units_1"], filter_data, sep,
                                              "score_descUnits", desc_unitsFileOut)
    #len(scored_desc_codelab)  # 4013114

    scored_desc_codelab  = variable_similarity(data, var_col, ["var_desc_1", "var_coding_labels_1"], filter_data, sep, "score_descCodeLab", desc_codelabFileOut)
    #len(scored_desc_codelab)  # 4013114

    scored_desc_codelab_units = variable_similarity(data, var_col, ["var_desc_1", "units_1", "var_coding_labels_1"], filter_data, sep, "score_descCodeLabUnits", allFileOut)
    #len(scored_full) #scored_desc_lab

    # Merge all scores files and write to merged file- CURRENTLY "SCORED" data frame is not returned from score_variables-so merged code below will not work with this code.
    # ##############################################################################
    #scored_merged = merge_score_results(scored, scored_codelab, "outer")
    #scored_merged = merge_score_results(scored_merged, scored_units, "outer")
    #scored_merged = merge_score_results(scored_merged, scored_desc_units, "outer")
    #scored_merged = merge_score_results(scored_merged, scored_desc_codelab, "outer")
    #scored_merged = merge_score_results(scored_merged, scored_desc_codelab_units, "outer")

    #scored_merged.to_csv("tiff_laura_shared/NLP text Score "
     #                     "results/All_Study_Similar_Variables_filtered_MERGED_7.17.19.csv", sep=",",
     #                     encoding="utf-8", index=False, line_terminator="\n")
    #
    # list(scored_merged)
    # len(scored_merged)
    #
    # # look at missing data by column
    # null_columns = scored_merged.columns[scored_merged.isnull().any()]
    # scored_merged[null_columns].isnull().sum()


if __name__ == "__main__":
    main()


