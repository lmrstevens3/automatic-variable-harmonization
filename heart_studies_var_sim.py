#!/usr/bin/env python
# coding: utf-8

# In[1]:


from automatic_variable_mapping import corpus, vocab_similarity
import pandas as pd
from functools import partial
import time
import numpy as np


# In[3]:

def default_pairable(score, pair_id, ref_ids, ref_id):
    return score >= 0 and pair_id != ref_id


def matching_groups(data, group_col, corpus_doc_ids, pair_id, ref_id):
    ref_idx = corpus_doc_ids.index(ref_id)
    pair_idx = corpus_doc_ids.index(pair_id)
    return data[pair_idx][group_col] != data[ref_idx][group_col]


def my_pairable(data, group_col, corpus_doc_ids, score, pair_id, _, ref_id):
    return default_pairable(score, pair_id, None, ref_id) and matching_groups(data, group_col, corpus_doc_ids, pair_id,
                                                                              ref_id)


def calc_score_results(data_file, doc_cols, ref_id_col, filter_file):
    data = pd.read_csv(data_file,
                       sep=",",
                       quotechar='"',
                       na_values="",
                       low_memory=False)
    if (filter_file != data_file):
        filter_data = pd.read_csv(filter_file,
                                  sep=",",
                                  quotechar='"',
                                  na_values="",
                                  low_memory=False)
    else:
        filter_data = data
    corpora_data = partition(data, 'study_1')
    corpora = corpus.build_corpora([doc_cols], [data], ref_id_col)
    tfidf_matrix = corpus.calc_tfidf(corpora)
    corpus_doc_ids = [doc_id for doc_id, _ in corpora]
    scores = vocab_similarity.VariableSimilarityCalculator(data[ref_id_col],
                                                           filter_data[ref_id_col],
                                                           pairable=partial(my_pairable, data,
                                                                            "TODO figure out the grouping column",
                                                                            corpus_doc_ids))

    scores.init_cache()
    scores.score_variables(corpora[0], tfidf_matrix)


ref_id_col = 'varDocID_1'
doc_col = 'var_desc_1'
# doc_col = list("var_desc_1”, “units_1, “var_coding_counts_distribution_1)
data_file = "tests/FHS_CHS_ARIC_MESA_varDoc_dbGaP_NLP.csv"
filt_file = "tests/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv"
score_file = 'tests/test_var_similarity_scores_rank_data.csv'

# In[4]:


data = pd.read_csv(data_file,
                   sep=",",
                   quotechar='"',
                   na_values="",
                   low_memory=False)
data.columns

# In[ ]:


corpora = corpus.build_corpora([doc_col], [data], ref_id_col)

# In[5]:


tfidf_matrix = corpus.calc_tfidf(corpora)

# In[9]:


reload(vocab_similarity)

# In[10]:


v = vocab_similarity.VariableSimilarityCalculator(data[ref_id_col])
v.init_cache()
v.score_variables(corpora[0], tfidf_matrix)

# In[99]:


orig_out_file_name = "tests/orig_file_out.csv"
orig_data = pd.read_csv(orig_out_file_name)

# In[101]:


v.cache.shape

# In[133]:


comb = pd.merge(orig_data, v.cache, how='left', left_on=['metadataID_1', 'metadataID_2'],
                right_on=['reference var', 'paired var']).round(6)

# In[138]:


assert comb.loc[comb['score'] == comb['score_desc']][
           ["score_desc", "score", "reference var", "metadataID_1", "metadataID_2", "paired var"]].shape[0] == \
       orig_data.shape[0]

