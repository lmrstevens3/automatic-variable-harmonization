#!/usr/bin/env python
# coding: utf-8

from automatic_variable_mapping import corpus, vocab_similarity
import pandas as pd
from functools import partial
import time
import numpy as np


# In[3]:
from automatic_variable_mapping.vocab_similarity import default_pairable, partition


def matching_groups(data, group_col, corpora, pair_id, ref_id):
    corpus_doc_ids = [doc_id for doc_id, _ in corpora]
    ref_idx = corpus_doc_ids.index(ref_id)
    pair_idx = corpus_doc_ids.index(pair_id)
    return data[pair_idx][group_col] == data[ref_idx][group_col]


def pairable_by_group(data, group_col, corpus_doc_ids, score, pair_id, _, ref_id):
    return vocab_similarity.default_pairable(score, pair_id, None, ref_id) and not matching_groups(data, group_col, corpus_doc_ids, pair_id, ref_id)


def calc_score_results(data_file, doc_cols, ref_id_col, filter_file, mult_corpora=False, corpora_col=None):
    data = pd.read_csv(data_file,
                       sep=",",
                       quotechar='"',
                       na_values="",
                       low_memory=False)
    if mult_corpora:
        corpora_data = partition(data, corpora_col)
    else:
        corpora_data = [data]
    if filter_file != data_file:
        filter_data = pd.read_csv(filter_file,
                                  sep=",",
                                  quotechar='"',
                                  na_values="",
                                  low_memory=False)
    else:
        filter_data = data

    corpora = corpus.build_corpora([doc_cols], corpora_data, ref_id_col)
    tfidf_matrix = corpus.calc_tfidf(corpora)

    scores = vocab_similarity.VariableSimilarityCalculator(filter_data[ref_id_col],
                                                           pairable=default_pairable)

    scores.init_cache()
    if mult_corpora:
        scores.score_variables(corpora, tfidf_matrix)
    else:
        scores.score_variables(corpora[0], tfidf_matrix)
    return(scores.cache)


obs_data_file = "tiff_laura_shared/FHS_CHS_ARIC_MESA_dbGaP_var_doc_NLP.csv"
obs_man_file = "tiff_laura_shared/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv"
ref_id_col = 'varDocID_1'
doc_cols_inputs = {'desc': ['var_desc_1'],
                   'units': ['units_1'],
                   'coding': ['var_coding_counts_distribution_1'],
                   'desc_units': ['var_desc_1', 'units_1'],
                   'desc_coding': ['var_desc_1', 'units_1', 'var_coding_counts_distribution_1'],
                   'desc_units_coding': ['var_desc_1', 'units_1', 'var_coding_counts_distribution_1']}
def calc_scores_doc_cols(data_file, doc_cols_inputs, ref_id_col, filter_file, mult_corpora=False, corpora_col=None):
    scores_dfs = list()
    for key in doc_cols_inputs:
        score_name = "score_" + key
        scores_df = calc_score_results(data_file, doc_cols_inputs[key], ref_id_col, filter_file, mult_corpora, corpora_col)
        scores_df = scores_df.rename({'score': score_name}, axis=1)
        scores_dfs.append(scores_df)

    scores_merged = reduce(lambda left, right: pd.merge(left, right, on=[ref_id_col],
                                                    how='outer'), scores_dfs)
    return(scores_merged)

obs_scores_tfidf = calc_scores_doc_cols(obs_data_file, doc_cols_inputs, ref_id_col, obs_man_file)

obs_scores_tfmcdf = calc_scores_doc_cols(obs_data_file, doc_cols_inputs, ref_id_col, obs_man_file, mult_corpora=True, corpora_col='study_1')
#TO DO add standard and add same above but for clinical trials

#doc_col = list("var_desc_1”, “units_1", “var_coding_counts_distribution_1")
score_file = 'tests/test_var_similarity_scores_rank_data.csv'





orig_out_file_name = "tests/orig_file_out.csv"

comb = pd.merge(orig_data, v.cache, how='left', left_on=['metadataID_1', 'metadataID_2'],
                right_on=['reference var', 'paired var']).round(6)

assert comb.loc[comb['score'] == comb['score_desc']][
           ["score_desc", "score", "reference var", "metadataID_1", "metadataID_2", "paired var"]].shape[0] == \
       orig_data.shape[0]

