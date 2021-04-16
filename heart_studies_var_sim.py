#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

import pandas as pd
from gensim.models import KeyedVectors

from automatic_variable_mapping import corpus, vocab_similarity
from automatic_variable_mapping.vocab_similarity import default_pairable, partition


def matching_groups(data, group_col, corpora, pair_id, ref_id):
    corpus_doc_ids = [doc_id for doc_id, _ in corpora]
    ref_idx = corpus_doc_ids.index(ref_id)
    pair_idx = corpus_doc_ids.index(pair_id)
    return data[pair_idx][group_col] == data[ref_idx][group_col]


def pairable_by_group(data, group_col, corpus_doc_ids, score, pair_id, _, ref_id):
    return vocab_similarity.default_pairable(score, pair_id, None, ref_id) and not matching_groups(data, group_col,
                                                                                                   corpus_doc_ids,
                                                                                                   pair_id, ref_id)


def doc_similarity(data_file, doc_cols, _ref_id_col, filter_file=None, word_vectors=None, corpora_col=None):
    # load data
    data = pd.read_csv(data_file,
                       sep=",",
                       quotechar='"',
                       na_values="",
                       low_memory=False)
    if corpora_col is not None:
        corpora_data = partition(data, corpora_col)
    else:
        corpora_data = [data]
    if filter_file is not None:
        filter_data = pd.read_csv(filter_file,
                                  sep=",",
                                  quotechar='"',
                                  na_values="",
                                  low_memory=False)
    else:
        filter_data = data

    # build corpora and calculate box matrix
    print(" Building Corpora...")
    corpora = corpus.build_corpora(doc_cols, corpora_data, _ref_id_col)

    print(" Calculating BOW matrix...")
    vocab, bow_matrix = corpus.calc_tfidf(corpora)

    # calculate embeddings
    if word_vectors:
        print(" Calculating doc embeddings matrix...")
        doc_vectors = corpus.calc_doc_embeddings(bow_matrix.toarray(), vocab, word_vectors)
    else:
        doc_vectors = bow_matrix.toarray()

    # calculate similarity scores
    print(" Calculating Similarity Scores...")
    scores = vocab_similarity.VariableSimilarityCalculator(filter_data[_ref_id_col], pairable=default_pairable)
    doc_ids = list(set([doc_id for c in corpora for doc_id, _ in c]))
    result = scores.score_docs(doc_ids, doc_vectors)
    return result


def calc_scores_doc_cols(data_file, doc_cols_input, _ref_id_col, filter_file, word_vectors=None, corpora_col=None):
    scores_dfs = list()
    print "~~~~~~~~~Similarity Scoring for: ", os.path.basename(data_file), "~~~~~~~~~"
    start_time = time.time()
    for key in doc_cols_input:
        print "-- Calculating Score: ", key, " --"
        score_name = "score_" + key
        scores_df = doc_similarity(data_file, doc_cols_input[key], _ref_id_col, filter_file, word_vectors, corpora_col)
        scores_df = scores_df.rename({'score': score_name}, axis=1)
        scores_dfs.append(scores_df)
        print scores_df.shape[0], " Scored Pairings Returned"

    scores_merged = reduce(lambda left, right: pd.merge(left, right, on=["reference var", "paired var"],
                                                        how='outer'), scores_dfs)

    print "-- Total Run Time: %s seconds --" % (time.time() - start_time)
    return scores_merged


# observational study files and test cases for documentation columns to include in documents for similarity
obs_data_file = "tiff_laura_shared/FHS_CHS_ARIC_MESA_dbGaP_var_doc_NLP.csv"
obs_man_file = "tiff_laura_shared/manual_concept_var_mappings_dbGaP_obs_heart_studies_NLP.csv"
ref_id_col = 'var_doc_id'
doc_cols_obs = {'desc': ['variable_description'],
                'units': ['units'],
                'coding': ['variable_coding_counts_distribution'],
                'desc_units': ['variable_description', 'units'],
                'desc_coding': ['variable_description', 'units', 'variable_coding_counts_distribution'],
                'desc_units_coding': ['variable_description', 'units', 'variable_coding_counts_distribution']}

# clinical trial study files (ref_id_col same as observational studies above)
trials_data_file = "tiff_laura_shared/HF_clin_trials_var_doc_BioLINCC.csv"
trials_man_file = "tiff_laura_shared/manual_concept_var_mappings_HF_clin_trials_biolincc_NLP.csv"
doc_cols_trials = {'desc': ['variable_description']}

# bioWordVec embeddings
vec_file_dir = "~/Downloads/"
binary_file_name = vec_file_dir + "BioWordVec_PubMed_MIMICIII_d200.vec.bin"
biowordvec_embeddings = KeyedVectors.load_word2vec_format(binary_file_name, binary=True, limit=None)

# observational studies
# tfidf
obs_scores_tfidf = calc_scores_doc_cols(obs_data_file, doc_cols_obs, ref_id_col, obs_man_file)
obs_scores_tfidf.to_csv("tiff_laura_shared/var_similarity_scores_obs_heart_studies.csv")
# biowordvec embeddings with regular tfidf
obs_scores_embed = calc_scores_doc_cols(obs_data_file, doc_cols_obs, ref_id_col, obs_man_file,
                                        word_vectors=biowordvec_embeddings)
obs_scores_embed.to_csv("tiff_laura_shared/var_similarity_embeddings_scores_obs_heart_studies.csv")
# tfcidf
# obs_scores_tfcidf = calc_scores_doc_cols(obs_data_file, doc_cols_inputs, ref_id_col, obs_man_file,  corpora_col='study')


# clinical trial studies
# tfidf
trials_scores_tfidf = calc_scores_doc_cols(trials_data_file, doc_cols_trials, ref_id_col, trials_man_file)
trials_scores_tfidf.to_csv("tiff_laura_shared/var_similarity_scores_HF_clin_trials.csv")
# biowordvec embeddings with regular tfidf
trials_scores_embed = calc_scores_doc_cols(trials_data_file, doc_cols_trials, ref_id_col, trials_man_file,
                                           word_vectors=biowordvec_embeddings)
trials_scores_embed.to_csv("tiff_laura_shared/var_similarity_embeddings_scores_HF_clin_trials.csv")
# tfcidf
# obs_scores_tfcidf = calc_scores_doc_cols(obs_data_file, doc_cols_inputs, ref_id_col, obs_man_file,  corpora_col='study')


# TO DO: add standard


df = pd.read_csv(trials_man_file,
                 sep=",",
                 quotechar='"',
                 na_values="",
                 low_memory=False)

df2 = pd.read_csv(trials_data_file,
                  sep=",",
                  quotechar='"',
                  na_values="",
                  low_memory=False)

len(set(df2['var_doc_id']).symmetric_difference(set(df['var_doc_id'])))

df3 = pd.merge(df, df2, on=['study', 'dataset_id', 'variable_id', 'var_doc_id'], how="outer", )
df[df.variable_description.isnull()].describe()

df = df['dataset_id'].str.strip()
df = df.apply(lambda x: x.str.strip())
df2 = df2.apply(lambda x: x.str.strip())
df['var_doc_id'] = df[['study', 'dataset_id', 'variable_id']].agg('.'.join, axis=1).str.lower()
# df = df.drop_duplicates('var_doc_id')
df.to_csv(trials_man_file)
df2.to_csv(trials_data_file)

# doc_col = list("var_desc_1”, “units_1", “var_coding_counts_distribution_1")
score_file = 'tests/test_var_similarity_scores_rank_data.csv'

orig_out_file_name = "tests/orig_file_out.csv"
#
# comb = pd.merge(orig_data, v.cache, how='left', left_on=['metadataID_1', 'metadataID_2'],
#                 right_on=['reference var', 'paired var']).round(6)
#
# assert comb.loc[comb['score'] == comb['score_desc']][
#            ["score_desc", "score", "reference var", "metadataID_1", "metadataID_2", "paired var"]].shape[0] == \
#        orig_data.shape[0]
