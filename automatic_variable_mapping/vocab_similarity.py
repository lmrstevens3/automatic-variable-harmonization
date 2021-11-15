##########################################################################################
# vocab_similarity.py
# author: TJ Callahan, Laura Stevens, Harrison Pielke-Lombardo
# Purpose: script reads in a csv files of variable documentation including some or all of
#           descriptions, units, as well as coding labels and pairs all variables against all other
#           variables (except against themselves) and scores variable similarity in an attempt to
#          identify which variables, using the documentation, are the most similar.
# version 1.1.1
# python version: 2.7.13
# date: 07.20.2020
##########################################################################################

# read in needed libraries
import nltk
import pandas as pd
# noinspection PyProtectedMember
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import math

import multiprocessing as mp
from functools import partial
import tqdm

# Move to the script
nltk.download('stopwords')
nltk.download("wordnet")


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
    cosine_similarities = linear_kernel(tfidf_matrix[ref_var_index:ref_var_index + 1], tfidf_matrix).flatten()
    rel_var_indices = [i for i in cosine_similarities.argsort()[::-1] if i != ref_var_index]
    similar_variables = [(variable, cosine_similarities[variable]) for variable in rel_var_indices]

    return similar_variables

def identity(*args):
    return args


def scores_identity(scores):
    return scores


class VariableSimilarityCalculator:

    def __init__(self, ref_ids, pairable=identity, select_scores=scores_identity, score_cols=None):
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
        self.file_name = None

    def init_cache(self, file_name=None):
        if file_name:
            with open(file_name, "w") as f:
                f.write(",".join(self.score_cols))
                f.write("\n")

    def score_variables(self, corpus_doc_ids, tfidf, pair_ids=None, num_cpus=1, file_name=None):
        """
        The function iterates over the corpus and returns the top_n (as specified by user) most similar variables,
        with a score, for each variable as a pandas data frame.

        :return: pandas data frame of the top_n (as specified by user) results for each variable
        """

        print "Finding valid pair ids"
        if pair_ids is None:
            pair_ids = corpus_doc_ids

        corpus_pair_indices = [corpus_doc_ids.index(pair_id) for pair_id in pair_ids]

        print "Pair ids: " + str(len(corpus_pair_indices))

        print "Finding valid ref ids"
        corpus_ref_indices = [corpus_doc_ids.index(ref_id) for ref_id in self.ref_ids if ref_id in corpus_doc_ids]

        print "Ref ids: " + str(len(corpus_ref_indices))

        # n = number of refs to find matches for
        # m = size of embeddings
        # k = number of possible ref matches
        # pair tfidf has shape [n, m]
        # transposed ref tfidf has shape [m, k]
        # [n, m] X [m, k] = [n, k]

        print "Multiplying matrices"
        pair_sub_tfidf = tfidf[corpus_pair_indices].toarray()
        print "LHS: " + str(pair_sub_tfidf.shape)
        ref_sub_tfidf = tfidf[corpus_ref_indices].toarray()
        print "RHS: " + str(ref_sub_tfidf.shape)
        cosine_similarities = np.matmul(ref_sub_tfidf, pair_sub_tfidf.transpose())
        print "Sim Matrix: " + str(cosine_similarities.shape)

        p = mp.pool.ThreadPool(processes=num_cpus)

        chunk_size = int(math.ceil(float(len(corpus_ref_indices)) / num_cpus))
        print "Chunk size: " + str(chunk_size)
        cache = list(tqdm.tqdm(p.map(partial(score_finder_helper,
                                             self.ref_ids,
                                             self.pairable,
                                             self.score_cols,
                                             self.select_scores,
                                             corpus_doc_ids,
                                             corpus_pair_indices,
                                             cosine_similarities,
                                             file_name),
                                     enumerate(corpus_ref_indices),
                                     chunk_size),
                               total=len(corpus_ref_indices)))

        if file_name:
            cache = [y for x in cache for y in x]
            with open(file_name, "a") as f:
                for cols in cache:
                    #cols = [d[col] for col in self.score_cols]
                    f.write(",".join(cols))
                    f.write("\n")
        else:
            cache = [y for x in cache for y in x]
            result = pd.DataFrame(cache, columns=self.score_cols)
            return result

def score_finder_helper(ref_ids, pairable, score_cols, select_scores, corpus_doc_ids, corpus_pair_indices, cosine_similarities, file_name, idx):
    col_idx, corpus_ref_idx = idx
    ref_id = corpus_doc_ids[corpus_ref_idx]
    if col_idx % 100 == 0:
        print "Finding matches for " + str(col_idx) + ' ' + ref_id

    similarities_vec = cosine_similarities[col_idx]
                # TODO HPL: I need to find a way to map the variable names to scores in the similarities_vec
    ref_var_scores = [(corpus_doc_ids[corpus_pair_indices[row_idx]], similarities_vec[row_idx])
                      for row_idx in similarities_vec.argsort()[::-1]]
    ref_var_scores = select_scores(ref_var_scores)
    ref_var_scores = filter_scores(ref_ids, pairable, ref_var_scores, ref_id)
    return cache_sim_scores(score_cols, ref_id, ref_var_scores, file_name)

def select_top_sims(similarities, n):
    # TODO HPL: This probably shouldn't be using a pandas dataframe
    return similarities[:n]

def cache_sim_scores(score_cols, ref_id, ref_var_scores, file_name=None):
    # retrieve top_n pairings for reference
    return [append_cache(score_cols, ref_id, pair_id, score, file_name=file_name) for pair_id, score in ref_var_scores]


def append_cache(score_cols, ref_doc_id, paired_doc_id, score, file_name=None):
    data = [ref_doc_id, paired_doc_id, str(score)]
    if file_name:
        return data
        # with open(file_name, "a") as f:
        #     f.write(",".join(data))
        #     f.write("\n")
    else:
        return dict(zip(score_cols, data))


def filter_scores(ref_ids, pairable, ref_var_scores, ref_id):
    return ((pair_id, score)
            for pair_id, score in ref_var_scores
            if pairable(score, pair_id, ref_ids, ref_id))


def helper(score_cols, ref_ids, pairable, select_scores, corpus_doc_ids, tfidf, c, ref_id):
    ref_id = str(ref_id)
    # get index of filter data in corpus
    corpus_ref_idx = corpus_doc_ids.index(ref_id)
    if corpus_ref_idx >= 0:
        ref_var_scores = calculate_similarity(tfidf, corpus_ref_idx)
        ref_var_scores = select_scores(ref_var_scores)
        ref_var_scores = filter_scores(ref_ids, pairable, ref_var_scores, ref_id)
        return cache_sim_scores(score_cols, c, ref_id, ref_var_scores)


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


def default_pairable(score, pair_id, ref_ids, ref_id):
    return score > 0 and pair_id != ref_id


def vals_differ_in_col(col):
    return lambda s1, s1_idx, s2, s2_idx: s1[col][s1_idx] != s2[col][s2_idx]


def vals_differ_in_all_cols(cols):
    return lambda s1, s1_idx, s2, s2_idx: all([s1[col][s1_idx] != s2[col][s2_idx] for col in cols])


def val_in_any_row_for_col(col):
    return lambda s1, s1_idx, s2, _: s1[col][s1_idx] in s2[col]




def select_top_sims_by_group(similarities, n, data, group_col):
    similarities.append(data[group_col])
    return similarities.sort_values(["score"], ascending=False).groupby(by=[group_col]).take(n)


def main():
    pass
    # dropbox_dir = "/Users/laurastevens/Dropbox/Graduate School/Data and MetaData Integration/ExtractMetaData/"
    # metadata_all_vars_file_path = dropbox_dir + "tiff_laura_shared/FCAMD_var_report_NLP_missing_contVars.csv" \
    #                                             "FHS_CHS_ARIC_MESA_dbGaP_var_report_dict_xml_Info_contVarNA_NLP_timeInterval_noDate_noFU_5-9-19.csv"
    # concept_mapped_vars_file_path = dropbox_dir + "CorrectConceptVariablesMapped_contVarNA_NLP.csv"
    #
    # # READ IN DATA -- 07.17.19
    # data = pd.read_csv(metadata_all_vars_file_path, sep=",", quotechar='"', na_values="",
    #                    low_memory=False)  # when reading in data, check to see if there is "\r" if
    # # not then don't use "lineterminator='\n'", otherwise u
    # data.units_1 = data.units_1.fillna("")
    # data.dbGaP_dataset_label_1 = data.dbGaP_dataset_label_1.fillna("")
    # data.var_desc_1 = data.var_desc_1.fillna("")
    # data.var_coding_labels_1 = data.var_coding_labels_1.fillna("")
    # len(data)
    #
    # # read in filtering file
    # filter_data = pd.read_csv(concept_mapped_vars_file_path, sep=",", na_values="", low_memory=False)  # n=700
    # filter_data.units_1 = filter_data.units_1.fillna("")
    # filter_data.dbGaP_dataset_label_1 = filter_data.dbGaP_dataset_label_1.fillna("")
    # filter_data.var_desc_1 = filter_data.var_desc_1.fillna("")
    # filter_data.var_coding_labels_1 = filter_data.var_coding_labels_1.fillna("")
    # len(filter_data)
    #
    # # CODE TO GENERATE RANDOM IDS
    # # data["random_id"] = random.sample(range(500000000), len(data))
    # # filter_data_m = filter_data.merge(data[['concat', 'random_id']], on='concat', how='inner').reset_index(drop=True)
    # # filter_data_m.to_csv("CorrectConceptVariablesMapped_RandomID_12.02.18.csv", sep=",", encoding="utf-8",
    # #                      index = False)
    #
    # id_col = "varDocID_1"
    #
    # save_dir = "tiff_laura_shared/NLP text Score results/"
    # # file_name_format = save_dir + "FHS_CHS_MESA_ARIC_text_similarity_scores_%s_ManuallyMappedConceptVars_7.17.19.csv"
    # # ref_suffix = "_1"
    # # pairing_suffix = "_2"
    # # score_cols = [id_col + ref_suffix,
    # #                 id_col.replace(pairing_suffix, "") + pairing_suffix]
    # file_name_format = save_dir + "test_%s_vocab_similarity.csv"
    # disjoint_col = 'dbGaP_studyID_datasetID_1'
    #
    # # data_cols_to_keep = ["study_1", 'dbGaP_studyID_datasetID_1', 'dbGaP_dataset_label_1', "varID_1",
    # #                     'var_desc_1', 'timeIntervalDbGaP_1', 'cohort_dbGaP_1']
    #
    # def my_pred(score, s1, i1, s2, i2):
    #     bools = [f(s1, i1, s2, i2)
    #              for f in [val_in_any_row_for_col(disjoint_col),
    #                        vals_differ_in_col(disjoint_col)]]
    #     bools.append(score > 0)
    #     return all(bools)
    #
    # top_n = len(data) - 1
    #
    # calc = VariableSimilarityCalculator(filter_data[id_col],
    #                                     pairable=my_pred,
    #                                     select_scores=lambda sims: select_top_sims_by_group(sims, top_n, data,
    #                                                                                         disjoint_col))
    #
    # score_name = "score_desc"
    # calc.score_cols[2] = score_name
    # file_name = file_name_format % "descOnly"
    # doc_col = ["var_desc_1"]
    # corpus_col = "study_1"
    # corpus_builder = CorpusBuilder(doc_col)
    # corpus_builder.build_corpus(partition(data, by=corpus_col), id_col)
    # corpus_builder.calc_tfidf()
    #
    # print '\n%s tfidf_matrix size %s' % (score_name, str(corpus_builder.tfidf_matrix.shape))
    #
    # calc.init_cache(file_name)
    #
    # scored = calc.score_variables(corpus_builder.all_docs(), corpus_builder.tfidf_matrix)
    # # scored = calc.variable_similarity(file_name, score_name, doc_col)
    # len(scored)  # 4013114

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
