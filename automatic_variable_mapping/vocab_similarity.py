##########################################################################################
# vocab_similarity.py
# author: Laura Stevens, TJ Callahan, Harrison Pielke-Lombardo
# Purpose: Creates a VariableSimilarityCalculator class and supporting functions to calculate cosine similarity for all
#          pairs of variable, or document, vectors against each other in an attempt to identify which variables,
#          or documents, are the most similar. The pairable functions specify which variables, or documents,
#          should not be paired and scored. Select_scores functions allow for filtering of scores results, such as
#          selecting a top set of scores or scores above a certain threshold.
# version 1.1.2
# python version: 2.7.13
# date: May 1, 2021
##########################################################################################

# read in needed libraries
import numpy as np
import pandas as pd
# noinspection PyProtectedMember
from sklearn.metrics.pairwise import linear_kernel
from itertools import izip
import warnings


def calculate_similarity(tfidf_matrix, ref_doc_index):
    """
    If the VariableSimilarityCalculator is instantiated, matrix multiplication is used instead of this function.
    The function calculates the cosine similarity between the index variables and all other included variables in
    the matrix. The results are sorted and returned as a list of lists, where each list contains a document
    identifier and the cosine similarity score for the set of paired documents are returned.

    :param tfidf_matrix:
    :param ref_doc_index: an integer representing a document id
    :return: a tuple with a ref doc index and a list of scored pairing tuples, each containing a paired doc index and
    the similarity score for the tuple's paired doc index and the ref doc index
    """

    # calculate similarity
    # cosine_similarity ==  linear_kernel(m1,m2) because tfidf matrix is l2 normalized,unit vectors by default
    cosine_similarities = linear_kernel(tfidf_matrix[ref_doc_index:ref_doc_index + 1], tfidf_matrix).flatten()
    paired_doc_indices = [i for i in cosine_similarities.argsort()[::-1] if i != ref_doc_index]
    similar_variables = [(paired_idx, cosine_similarities[paired_idx]) for paired_idx in paired_doc_indices]
    return similar_variables


def identity(*args):
    return args


def scores_identity(scores):
    return scores


class VariableSimilarityCalculator:

    def __init__(self, ref_ids, pairable=identity, select_scores=scores_identity, score_cols=None):
        """

        :param score_cols: column names that should be used in the cache or the results returned
        :param select_scores: a function to return a selection of ids for each ref_id input (ex. top n scored pairings)
        :param ref_ids: an iterable containing reference, document ids in a corpora that should be paired and scored
        against all other documents

        """
        if not score_cols:
            score_cols = ["reference_id", "paired_id", "score"]
        self.ref_ids = ref_ids
        self.pairable = pairable
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

    def score_variables(self, corpora, tfidf, pair_ids=None, num_cpus=None):
        """
        The function iterates over the corpus and returns the top_n (as specified by user) most similar variables,
        with a score, for each variable as a pandas data frame.
        with a score, for each variable as a pandas data frame.

     Given a set of reference, doc_ids in a corpora, the cosine similarities for each reference, doc paired with
         all other docs in doc_vectors is calculated for all reference docs. The pairings returned are then filtered
         based on pairable and select_scores functions specified. The function implements the following:

        1. Consolidate all ref_ids that need to be mapped
        2. Subset doc_vectors matrix accordingly
        3. In a single matrix multiplication operation, calculate cosine similarity between proposed doc_vectors sub-matrix and the whole doc_vectors matrix
        4. Re-map ref_ids to cosine-similarity matrix and return pairing results

        :param doc_ids: a list of doc ids for the documents represented in the doc_vectors matrix
        :param doc_vectors: a vector representation of documents (e.g. bow matrix or embedding matrix)
        :param cache: Either a file or, if left unspecified, a list
        :return: If cache file is specified, results are written to a file, otherwise a pandas data frame is returned,
         with reference_id (id of reference doc), paired_id (id of the paired doc), similarity score
        """

        # n = number of refs to find matches for
        # m = size of embeddings
        # k = number of possible ref matches
        # length n
        # [n, m]
        if doc_vectors.shape[0] != len(doc_ids):
            doc_ids = doc_ids[0:doc_vectors.shape[0]]
            warnings.warn("doc_ids and doc_vectors length aren't equal, only ref_ids/doc_ids with doc vectors are scored")

        ref_id_indices = [doc_ids.index(ref_id) for ref_id in self.ref_ids if ref_id in doc_ids]
        ref_id_indices.sort()
        sub_vectors = doc_vectors[ref_id_indices]
        # doc_vectors has shape [m, k]
        # [n, m] X [m, k] = [n, k]
        # cosine similarity = [n, m] X [m, k] (dot product), because tfidf matrix is l2 normalized (unit vectors) by default
        # print "   Calculating Similarity Scores"
        cosine_similarities = np.matmul(doc_vectors, sub_vectors.transpose())

        if cache:
            with open(cache, "w") as f:
                f.write(",".join(self.score_cols)+"\n")
        else:
            cache = []

        # print "   Getting Pairings for Ref IDs"
        for ref_id_idx, similarities in izip(ref_id_indices, cosine_similarities.transpose()):
            # print "Finding matches for", ref_doc_idx
            ref_id = doc_ids[ref_id_idx]
            paired_doc_indices = [i for i in similarities.argsort()[::-1] if i != ref_id_idx]
            ref_doc_scores = [(doc_ids[paired_doc_idx], similarities[paired_doc_idx]) for paired_doc_idx in paired_doc_indices]
            ref_doc_scores = self.select_scores(ref_doc_scores)
            ref_doc_scores = filter_scores(self.pairable, ref_doc_scores, ref_id)
            cache_sim_scores(cache, self.score_cols, ref_id, ref_doc_scores)
        if isinstance(cache, list):
            cache = pd.DataFrame(cache, columns=self.score_cols)
            return cache


def cache_sim_scores(cache, score_cols, ref_id, ref_doc_scores):
    # add all pairing scores for ref_id of reference doc
    for paired_id, score in ref_doc_scores:
        data = [ref_id, paired_id, score]
        if isinstance(cache, list):
            cache.append(dict(zip(score_cols, data)))
        else:
            # Assume cache is a file to write to.
            with open(cache, "a") as f:
                f.write(",".join([str(val) for val in data])+"\n")


def filter_scores(pairable, ref_doc_scores, ref_id):
    return ((paired_id, score) for paired_id, score in ref_doc_scores
            if pairable(score, paired_id, ref_id))


# default pairable filters pairings with a score of 0 and self-pairings
def default_pairable(score, paired_id, ref_id):
    # checking paired_id != ref_id is checked in score_docs when
    return score > 0 and paired_id != ref_id


# additional pairable functions to be used with or instead of default_pairable
#   to use these functions, use partial from functools library. For example:
#   pairable=partial(pairable_groups_disjoint, data=data, id_col = 'id', group_cols = 'group')
def matching_groups(data, group_cols, id_col, pair_id, ref_id):
    # check if pairings are part of the same class/group col in data used for corpora for one or multiple groups
    ref_idx = data.loc[data[id_col] == ref_id].index[0]
    pair_idx = data.loc[data[id_col] == pair_id].index[0]
    return all([data[pair_idx][group_col] == data[ref_idx][group_col] for group_col in group_cols])


def pairable_groups_disjoint(data, group_cols, id_col, score, pair_id, ref_id):
    # check that pairings are not part of the same class/group col for one or multiple groups
    return default_pairable(score, pair_id, ref_id) and not matching_groups(data, group_cols, id_col, pair_id, ref_id)


def pairable_groups_intersect(data, group_cols, id_col, score, pair_id, ref_id):
    # check that pairings are not part of the same class/group col for one or multiple groups
    return default_pairable(score, pair_id, ref_id) and matching_groups(data, group_cols, id_col, pair_id, ref_id)


# select scores returns all scores by default, top n scores, or topn scores by group can use functions below
            cache = [y for x in cache for y in x]
        else:
            # TODO HPL:
            # run on GPU
            # 1. Consolidate all ref_ids that need to be mapped
            # 2. Subset tfidf matrix accordingly
            # 3. In a single matrix multiplication operation, calculate cosine similarity between proposed tfidf submatrix and the whole tfidf matrix
            # 4. Re-map ref_ids to cosine-similarity matrix

            print "Finding valid pair ids"

            if pair_ids is None:
                pair_ids = corpus_doc_ids

            corpus_pair_indices = list()
            for pair_id in pair_ids:
                pair_id = str(pair_id)
                # get index of filter data in corpus
                corpus_pair_idx = corpus_doc_ids.index(pair_id)
                if corpus_pair_idx >= 0:
                    corpus_pair_indices.append(corpus_pair_idx)

            print "Pair ids: " + str(len(corpus_pair_indices))

            print "Finding valid ref ids"
            corpus_ref_indices = list()
            for ref_id in self.ref_ids:
                ref_id = str(ref_id)
                # get index of filter data in corpus
                corpus_ref_idx = corpus_doc_ids.index(ref_id)
                if corpus_ref_idx >= 0:
                    corpus_ref_indices.append(corpus_ref_idx)

            print "Ref ids: " + str(len(corpus_ref_indices))

            # n = number of refs to find matches for
            # m = size of embeddings
            # k = number of possible ref matches
            # pair tfiidf has shape [n, m]
            # transposed ref tfidf has shape [m, k]
            # [n, m] X [m, k] = [n, k]

            print "Multiplying matrices"
            pair_sub_tfidf = tfidf[corpus_pair_indices].toarray()
            print "LHS: " + str(pair_sub_tfidf.shape)
            ref_sub_tfidf = tfidf[corpus_ref_indices].toarray()
            print "RHS: " + str(ref_sub_tfidf.shape)
            cosine_similarities = np.matmul(ref_sub_tfidf, pair_sub_tfidf.transpose())
            print "Sim Matrix: " + str(cosine_similarities.shape)

            cache = list()
            for col_idx, corpus_ref_idx in enumerate(corpus_ref_indices):
                ref_id = corpus_doc_ids[corpus_ref_idx]
                print "Finding matches for", col_idx, corpus_ref_idx, ref_id
                similarities_vec = cosine_similarities[col_idx]
                # TODO HPL: I need to find a way to map the variable names to scores in the similarities_vec
                ref_var_scores = [(corpus_doc_ids[corpus_pair_indices[row_idx]], similarities_vec[row_idx])
                                  for row_idx in similarities_vec.argsort()[::-1]
                                  if corpus_doc_ids[corpus_pair_indices[row_idx]] != ref_id]
                ref_var_scores = self.select_scores(ref_var_scores)
                ref_var_scores = filter_scores(self.ref_ids, self.pairable, ref_var_scores, ref_id)
                cache.append(cache_sim_scores(self.score_cols, ref_id, ref_var_scores))

            cache = [y for x in cache for y in x]

        result = pd.DataFrame(cache, columns=self.score_cols)

        return result

def select_top_sims(similarities, n):
    # TODO HPL: This probably shouldn't be using a pandas dataframe
    return similarities[:n]

def cache_sim_scores(score_cols, ref_id, ref_var_scores):
    # retrieve top_n pairings for reference
    return [append_cache(score_cols, ref_id, pair_id, score) for pair_id, score in ref_var_scores]


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

def default_pairable(score, pair_id, ref_ids, ref_id):
    return score >= 0 and pair_id != ref_id


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
    # READ IN DATA -- 07.17.19
    testData = pd.read_csv(varDocFile, sep=",", quotechar='"', na_values="",
                           low_memory=False)  # when reading in data, check
    #  to see if there is "\r" if # not then don't use "lineterminator='\n'", otherwise u

# select top pairings by a particular class/group
def select_top_sims_by_group(ref_doc_scores, n, id_group_dict):
    scores_grps = [(paired_id, score, id_group_dict[paired_id]) for paired_id, score in ref_doc_scores
                   if paired_id in id_group_dict.keys()]
    scores_groups = pd.DataFrame(scores_grps, columns=['paired_idx', 'score', 'group'])
    top_scores_groups = scores_groups.sort_values(['group', 'score'], ascending=False).groupby(by='group').head(n)
    return top_scores_groups[['paired_idx', 'score']].to_records(index=False).tolist()
