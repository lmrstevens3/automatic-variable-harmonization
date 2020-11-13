#!/usr/bin/env python
# coding: utf-8

# In[1]:


from automatic_variable_mapping import corpus, vocab_similarity
import pandas as pd
import time
import numpy as np


# In[3]:


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


comb = pd.merge(orig_data, v.cache, how='left', left_on=[ 'metadataID_1', 'metadataID_2' ], right_on=[ 'reference var', 'paired var' ]).round(6)


# In[138]:


assert comb.loc[comb['score'] == comb['score_desc']][["score_desc", "score", "reference var", "metadataID_1", "metadataID_2", "paired var"]].shape[0] == orig_data.shape[0]

