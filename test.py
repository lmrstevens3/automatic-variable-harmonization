from automatic_variable_mapping import corpus, vocab_similarity
from automatic_variable_mapping.vocab_similarity import VariableSimilarityCalculator
import pandas as pd
import numpy as np

ref_id_col = 'dbGaP_studyID_datasetID_varID_1'
doc_col = 'var_desc_1'
# doc_col = list("var_desc_1”, “units_1, “var_coding_counts_distribution_1)
data_file = "tests/test_mannual_map_ref_data.csv"
score_file = 'tests/test_var_similarity_scores_rank_data.csv'

data = pd.read_csv(data_file,
                      sep=",",
                      quotechar='"',
                      na_values="",
                      low_memory=False)

corpora = corpus.build_corpora([doc_col], [data], ref_id_col)

tfidf_matrix = corpus.calc_tfidf(corpora)

v = vocab_similarity.VariableSimilarityCalculator(data[ref_id_col])
v.init_cache()
result = v.score_docs(corpora, tfidf_matrix)


corpus_doc_ids = [doc_id for doc_id, _ in corpora]
corpus_ref_idx_to_ref_id = {}
for ref_id in v.ref_ids:
    ref_id = str(ref_id)
    # get index of filter data in corpus
    corpus_ref_idx = corpus_doc_ids.index(ref_id)
    if corpus_ref_idx >= 0:
        corpus_ref_idx_to_ref_id[corpus_ref_idx] = ref_id

# n = number of refs to find matches for
# m = size of embeddings
# k = number of possible ref matches
# length n
ids_to_subset = list(set(corpus_ref_idx_to_ref_id.keys()))
# [n, m]
sub_tfidf = tfidf_matrix[ids_to_subset]
# tfidf has shape [m, k]
# [n, m] X [m, k] = [n, k]
cosine_similarities = np.matmul(tfidf_matrix.toarray(),
                                sub_tfidf.toarray().transpose())


def select_top_sims(similarities, n):
    # TODO HPL: This probably shouldn't be using a pandas dataframe
    return similarities[:n]

v.select_scores = lambda similarities: select_top_sims(similarities, 5)


def cache_sim_scores(score_cols, c, ref_id, ref_var_scores):
    # retrieve top_n pairings for reference
    return [append_cache(score_cols, ref_id, c[i][0], score) for i, score in ref_var_scores]


def append_cache(score_cols, ref_doc_id, paired_doc_id, score, file_name=None):
    data = [ref_doc_id, paired_doc_id, score]
    if file_name:
        with open(file_name, "a") as f:
            f.write(",".join(data))
            f.write("\n")
    else:
        return dict(zip(score_cols, data))

# ###
i = 0
corpus_ref_idx = ids_to_subset[i]
similarities_vec = cosine_similarities[i]

rel_var_indices = [i for i in similarities_vec.argsort()[::-1] if i != corpus_ref_idx]

ref_var_scores1 = [(variable, similarities_vec[variable]) for variable in rel_var_indices]
ref_var_scores2 = v.select_scores(ref_var_scores1)

ref_var_scores3 = list(vocab_similarity.filter_scores(v.ref_ids, v.pairable, ref_var_scores2, ref_id))


print(ref_var_scores3[0])
[append_cache(v.score_cols, ref_id, corpora[i][0], score) for i, score in ref_var_scores3]
cache.append(vocab_similarity.cache_sim_scores(v.score_cols, corpora, ref_id, ref_var_scores3))
# ###

cache = list()
for i, corpus_ref_idx in enumerate(ids_to_subset):
    similarities_vec = cosine_similarities[i]
    ref_id = corpus_ref_idx_to_ref_id[corpus_ref_idx]
    rel_var_indices = [i for i in similarities_vec.argsort()[::-1] if i != corpus_ref_idx]
    ref_var_scores = [(variable, similarities_vec[variable]) for variable in rel_var_indices]
    # TODO HPL: Need to fix select_scores
    ref_var_scores = v.select_scores(ref_var_scores)
    ref_var_scores = vocab_similarity.filter_scores(v.ref_ids, v.pairable, ref_var_scores, ref_id)
    cache.append(vocab_similarity.cache_sim_scores(v.score_cols, corpora, ref_id, ref_var_scores))

cache2 = [y for x in cache for y in x]

result = pd.DataFrame(cache2, columns=v.score_cols)
