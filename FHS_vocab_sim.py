#!/usr/bin/env python
# coding: utf-8

# In[1]:


from automatic_variable_mapping import corpus
from automatic_variable_mapping.vocab_similarity import VariableSimilarityCalculator
import pandas as pd
import numpy as np
import multiprocessing


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

filt_data_file = "tests/manualConceptVariableMappings_dbGaP_Aim1_contVarNA_NLP.csv"
filt_data = pd.read_csv(filt_data_file,
                      sep=",",
                      quotechar='"',
                      na_values="",
                      low_memory=False)

# In[4]:


corpora = corpus.build_corpora([doc_col], [data], ref_id_col, num_cpus=multiprocessing.cpu_count()-5)


# In[10]:


tfidf_matrix = corpus.calc_tfidf(corpora)


# In[12]:


v = VariableSimilarityCalculator(filt_data[ref_id_col])
v.init_cache()
results = v.score_variables(corpora[0], tfidf_matrix, num_cpus=multiprocessing.cpu_count()-5)

pd.write_csv(results, "results.csv")
