from gensim.models import KeyedVectors
import numpy as np

def convert_to_keyedvectors(input_file_name, output_file_name):
    #loads embedding file specified from word2vec format and saves them as keyed vectors in .vec format
    model = KeyedVectors.load_word2vec_format(input_file_name,
                                              binary=True,
                                              limit=None)
    model.save(output_file_name)



def load_keyed_vectors(input_file_name):
    #loads .vec keyed vectors and maps memory over all CPUS
    return KeyedVectors.load(input_file_name, mmap='r')


vec_file_dir = "~/Downloads/"
binary_file_name = vec_file_dir  + "BioWordVec_PubMed_MIMICIII_d200.vec.bin"
#binary_model_file_name = vec_file_dir  + "BioWordVec_PubMed_MIMICIII_d200.bin"
#vector_file_name =  vec_file_dir  + "bio-word-vectors.vec"


#convert_to_keyedvectors(binary_file_name, vector_file_name) #wouldn't save .vec file on full embedding file (limit = None), so didn't use load_keyed_vectors
word_embeddings = KeyedVectors.load_word2vec_format(binary_file_name, binary=True, limit=10)


#develop embedding pipeline for doc sentences
from automatic_variable_mapping import corpus, vocab_similarity
from automatic_variable_mapping.vocab_similarity import default_pairable, partition
import pandas as pd
import time
f =  "tiff_laura_shared/manual_concept_var_mappings_dbGaP_obs_heart_studies_NLP.csv"
id_col = 'var_doc_id'
doc_col = ['var_desc']
df = pd.read_csv(f,
                 sep=",",
                 quotechar='"',
                 na_values="",
                 low_memory=False)

corpora = corpus.build_corpora(doc_col, [df], id_col)
vocab, bow_matrix = corpus.calc_tfidf(corpora)

embedding_matrix = [word_embeddings[word] if word in word_embeddings else np.zeros(word_embeddings.vector_size) for word in vocab]
doc_embeddings = np.matmul(bow_matrix.toarray(), np.array(embedding_matrix))

scores = vocab_similarity.VariableSimilarityCalculator(df[id_col], pairable=default_pairable)
time.time()
result = scores.score_docs(corpora, doc_embeddings)
time.time()