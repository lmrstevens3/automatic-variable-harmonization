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
from progressbar import ProgressBar, FormatLabel, Percentage, Bar
# noinspection PyProtectedMember
from sklearn.metrics.pairwise import linear_kernel

import multiprocessing
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
    similarities = linear_kernel(tfidf_matrix[ref_var_index:ref_var_index + 1], tfidf_matrix)
    similarities = pd.DataFrame({'score': similarities[0], 'idx': range(0, len(similarities[0]))})
    similarities = similarities.loc[similarities['idx'] != ref_var_index]
    # cosine_similarities = ((i, score) for i, score in enumerate(cosine_similarities) if i != ref_var_index)

    return [(int(i), score) for i, score in similarities.values]


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
        self.cache = None
        self.file_name = None

    def init_cache(self, file_name=None):
        self.file_name = file_name
        self.cache = pd.DataFrame([], columns=list(self.score_cols))
        if self.file_name:
            with open(self.file_name, "w") as f:
                f.write(",".join(self.score_cols))
                f.write("\n")

    # def finalize_cached_output(self):
    #     if not self.file_name:
    #         self.cache.to_csv(self.file_name, sep=",", encoding="utf-8", index=False, line_terminator="\n")
    #     print '\n' + self.file_name + " written"  # " scored size:" + str(len(scored))  # 4013114

    def score_variables(self, c, tfidf, num_cpus=None):
        """
        The function iterates over the corpus and returns the top_n (as specified by user) most similar variables,
        with a score, for each variable as a pandas data frame.

        :return: pandas data frame of the top_n (as specified by user) results for each variable
        """

        # cache = init_cache_output(None, "pandas", file_name)

        # matching data in filtered file

        # tqdm.pandas(desc="Filtering the data")
        # filter_data.progress_apply(lambda row: ,axis=1)

        corpus_doc_ids = [doc_id for doc_id, _ in c]

        p = multiprocessing.Pool(processes=num_cpus)

        cache = list(tqdm.tqdm(p.imap(partial(helper,
                                                self.score_cols,
                                                self.ref_ids,
                                                self.pairable,
                                                self.select_scores,
                                                corpus_doc_ids,
                                                tfidf,
                                                c),
                                      self.ref_ids),
                               total=len(self.ref_ids)))

        result = pd.DataFrame(columns=self.score_cols)
        cache = [y for x in cache for y in x]
        for y in cache:
            result = result.append(y, ignore_index=True)
        # pbar.finish()

        # verify that we got all the matches we expected (assumes that we should be able to
        # match all vars in filtered data)

        # if matches != len(self.ref_ids):
        #     matched = round(matches / float(len(self.ref_ids)) * 100, 2)
        #     raise ValueError('There is a problem - Only matched {0}% of filtered variables'.format(matched))

        # self.finalize_cached_output()

        # print("Filtering matched " + str(matches) + " of " + str(len(self.ref_ids)) + " variables")
        return result

    # def variable_similarity(self, file_name, score_name, doc_col, data, id_col):
    #     # PRE-PROCESS DATA & BUILD CORPORA
    #     # var_col and defn/units/codeLabels_col hold information from the data frame and are used when
    #     # processing the data
    #
    #     corpus_builder = CorpusBuilder(doc_col)
    #     corpus_builder.build_corpus(data, id_col)
    #     corpus_builder.calc_tfidf()
    #     print '\n' + score_name + " tfidf_matrix size:"
    #     print corpus_builder.tfidf_matrix.shape  # 105611 variables and 33031 unique concepts
    #
    #     self.init_cache(file_name)
    #
    #     # SCORE DATA + WRITE OUT RESULTS
    #     return self.score_variables(corpus_builder.all_docs(), corpus_builder.tfidf_matrix)


def cache_sim_scores(score_cols, c, ref_id, ref_var_scores):
    # retrieve top_n pairings for reference
    return [append_cache(score_cols, ref_id, c[i][0], score) for i, score in ref_var_scores]


def append_cache(score_cols, ref_doc_id, paired_doc_id, score, file_name=None):
    data = [ref_doc_id, paired_doc_id, score]
    if file_name:
        with open(file_name, "a") as f:
            f.write(",".join(data))
            f.write("\n")
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
