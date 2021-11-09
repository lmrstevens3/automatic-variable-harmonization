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

    def score_docs(self, doc_ids, doc_vectors, cache=None):
        """
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
def select_top_sims(ref_doc_scores, n):
    return ref_doc_scores[:n]


# select top pairings by a particular class/group
def select_top_sims_by_group(ref_doc_scores, n, id_group_dict):
    scores_grps = [(paired_id, score, id_group_dict[paired_id]) for paired_id, score in ref_doc_scores
                   if paired_id in id_group_dict.keys()]
    scores_groups = pd.DataFrame(scores_grps, columns=['paired_idx', 'score', 'group'])
    top_scores_groups = scores_groups.sort_values(['group', 'score'], ascending=False).groupby(by='group').head(n)
    return top_scores_groups[['paired_idx', 'score']].to_records(index=False).tolist()
