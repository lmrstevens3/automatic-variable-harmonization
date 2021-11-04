#!/usr/bin/env python
# coding: utf-8

# In[1]:


from automatic_variable_mapping import tfidf


# In[2]:


from automatic_variable_mapping import corpus, vocab_similarity
import pandas as pd
import time
import numpy as np


# In[3]:


# doc_col = list("var_desc_1”, “units_1, “var_coding_counts_distribution_1)
data_file = "SNOMED-concepts/output/combined_FHS_SNOMED.csv"
filt_file = "tests/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv"
score_file = 'tests/test_var_similarity_scores_rank_data.csv'


# In[4]:


data = pd.read_csv(data_file,
                      sep=",",
                      quotechar='"',
                      na_values="",
                      low_memory=False)
data


# In[5]:


# For testing reduce data size
data_orig = data


# In[11]:


data = data_orig


# In[12]:


ref_id_col = 'var_doc_id'
doc_col = 'variable_description'
corpora = corpus.build_corpora([doc_col], [data], ref_id_col)
len(corpora[0]), len(corpora[0][0])


# In[13]:


tfidf_matrix = corpus.calc_tfidf(corpora)
tfidf_matrix.shape


# In[14]:


#reload(vocab_similarity) 


# In[15]:


v = vocab_similarity.VariableSimilarityCalculator(data[ref_id_col])
v.init_cache()
scores = v.score_variables(corpora[0], tfidf_matrix)


# In[61]:


scores


# In[62]:


scores.to_csv("output/combined_dbGaP_SNOMED_sim_scores.csv")

