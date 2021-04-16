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

import numpy as np
import pandas as pd
# noinspection PyProtectedMember
from sklearn.metrics.pairwise import linear_kernel


# ssl certificates are failing on mac. Can check for files in ~/nltk_data and if not present then can try download or manually download and save to that location
# nltk.download('stopwords')
# nltk.download('wordnet')


def calculate_similarity(tfidf_matrix, ref_doc_index):
    """
    The function calculates the cosine similarity between the index variables and all other included variables in
    the matrix. The results are sorted and returned as a list of lists, where each list contains a variable
    identifier and the cosine similarity score for the top set of similar variables as indicated by the input
    argument are returned.

    :param tfidf_matrix:
    :param ref_doc_index: an integer representing a variable id
    :return: a list of lists where each list contains a variable identifier and the cosine similarity
        score the top set of similar as indicated by the input argument are returned
    """

    # calculate similarity
    cosine_similarities = linear_kernel(tfidf_matrix[ref_doc_index:ref_doc_index + 1], tfidf_matrix).flatten()
    rel_doc_indices = [i for i in cosine_similarities.argsort()[::-1] if i != ref_doc_index]
    similar_variables = [(variable, cosine_similarities[variable]) for variable in rel_doc_indices]

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
        self.cache = None
        self.file_name = None

    def score_docs(self, doc_ids, doc_vectors):
        """
        The function iterates over the corpus and returns the top_n (as specified by user) most similar variables,
        with a score, for each variable as a pandas data frame.

        1. Consolidate all ref_ids that need to be mapped
        2. Subset tfidf matrix accordingly
        3. In a single matrix multiplication operation, calculate cosine similarity between proposed tfidf submatrix and the whole tfidf matrix
        4. Re-map ref_ids to cosine-similarity matrix

        :return: pandas data frame of the top_n (as specified by user) results for each variable
        """

        # n = number of refs to find matches for
        # m = size of embeddings
        # k = number of possible ref matches
        # length n
        # [n, m]
        ref_id_indices = [doc_ids.index(ref_id) for ref_id in self.ref_ids]
        ref_id_indices.sort()
        sub_vectors = doc_vectors[ref_id_indices]
        # tfidf has shape [m, k]
        # [n, m] X [m, k] = [n, k]
        # print " Calculating Similarity Scores"
        cosine_similarities = np.matmul(doc_vectors, sub_vectors.transpose())

        cache = list()
        # print " Getting Pairings for Ref IDs"
        for ref_id_idx, similarities_vec in zip(ref_id_indices, cosine_similarities):
            # print "Finding matches for", ref_doc_idx
            ref_id = doc_ids[ref_id_idx]
            paired_doc_indices = [i for i in similarities_vec.argsort()[::-1] if i != ref_id_idx]
            ref_doc_scores = [(paired_doc_idx, similarities_vec[paired_doc_idx]) for paired_doc_idx in
                              paired_doc_indices]
            ref_doc_scores = self.select_scores(ref_doc_scores)
            ref_doc_scores = filter_scores(self.ref_ids, self.pairable, ref_doc_scores, ref_id)
            cache.append(cache_sim_scores(self.score_cols, doc_ids, ref_id, ref_doc_scores))

        cache = [y for x in cache for y in x]

        result = pd.DataFrame(cache, columns=self.score_cols)
        return result


def select_top_sims(similarities, n):
    # TODO HPL: This probably shouldn't be using a pandas dataframe
    return similarities[:n]


def cache_sim_scores(score_cols, doc_ids, ref_id, ref_doc_scores):
    # retrieve top_n pairings for reference
    return [append_cache(score_cols, ref_id, doc_ids[paired_doc_idx], score) for paired_doc_idx, score in
            ref_doc_scores]


def append_cache(score_cols, ref_doc_id, paired_id, score, file_name=None):
    data = [ref_doc_id, paired_id, score]
    if file_name:
        with open(file_name, "a") as f:
            f.write(",".join(data))
            f.write("\n")
    else:
        return dict(zip(score_cols, data))


def filter_scores(ref_ids, pairable, ref_doc_scores, ref_id):
    return ((pair_id, score)
            for pair_id, score in ref_doc_scores
            if pairable(score, pair_id, ref_ids, ref_id))


def helper(score_cols, ref_ids, pairable, select_scores, corpus_doc_ids, tfidf, c, ref_id):
    ref_id = str(ref_id)
    # get index of filter data in corpus
    corpus_ref_idx = corpus_doc_ids.index(ref_id)
    if corpus_ref_idx >= 0:
        ref_doc_scores = calculate_similarity(tfidf, corpus_ref_idx)
        ref_doc_scores = select_scores(ref_doc_scores)
        ref_doc_scores = filter_scores(ref_ids, pairable, ref_doc_scores, ref_id)
        return cache_sim_scores(score_cols, c, ref_id, ref_doc_scores)


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


if __name__ == "__main__":
    main()
