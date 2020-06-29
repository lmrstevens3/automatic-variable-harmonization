#!/usr/bin/env python
# coding: utf-8

# In[1]:


from automatic_variable_mapping import corpus
from automatic_variable_mapping.vocab_similarity import VariableSimilarityCalculator
import pandas as pd
import numpy as np


# In[2]:


ref_id_col = 'dbGaP_studyID_datasetID_1'
doc_col = 'var_desc_1'
# doc_col = list("var_desc_1”, “units_1, “var_coding_counts_distribution_1)
data_file = "tests/FHS_CHS_ARIC_MESA_varDoc_dbGaP_NLP.csv"
score_file = 'tests/test_var_similarity_scores_rank_data.csv'


# In[3]:


data = pd.read_csv(data_file,
                      sep=",",
                      quotechar='"',
                      na_values="",
                      low_memory=False)


# In[4]:


corpora = corpus.build_corpora([doc_col], [data], ref_id_col)


# In[10]:


tfidf_matrix = corpus.calc_tfidf(corpora)


# In[12]:


v = VariableSimilarityCalculator(data[ref_id_col])
v.init_cache()
v.score_variables(corpora[0], tfidf_matrix, num_cpus=3)


# In[99]:


orig_out_file_name = "tests/orig_file_out.csv"
orig_data = pd.read_csv(orig_out_file_name)


# In[101]:


v.cache.shape


# In[133]:


comb = pd.merge(orig_data, v.cache, how='left', left_on=[ 'metadataID_1', 'metadataID_2' ], right_on=[ 'reference var', 'paired var' ]).round(6)


# In[138]:


assert comb.loc[comb['score'] == comb['score_desc']][["score_desc", "score", "reference var", "metadataID_1", "metadataID_2", "paired var"]].shape[0] == orig_data.shape[0]

