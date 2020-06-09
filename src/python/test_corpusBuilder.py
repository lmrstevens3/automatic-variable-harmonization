import pandas as pd
from unittest import TestCase

from src.python.Vocab_Similarity import CorpusBuilder


class TestCorpusBuilder(TestCase):
    def test_calc_tfidf(self):
        file_path = "test_vocab_similarity_data.csv"

        # Read in test data and sample for test
        data = pd.read_csv(file_path, sep=",", quotechar='"', na_values="",  low_memory=False)
        data.var_desc_1 = data.var_desc_1.fillna("")

        id_col = "varDocID_1"

        nrows = 10
        data = [data.sample(nrows, random_state=22895)]

        doc_col = ["var_desc_1"]
        corpus_builder = CorpusBuilder(doc_col)
        corpus_builder.build_corpora(data, id_col)

        corpus_builder.calc_tfidf()

        assert corpus_builder.tfidf_matrix.shape == (nrows, 79)  # using
        assert corpus_builder.tfidf_matrix[0, 75].round(10) == 0.3692482208


    def test_multi_corpus_tfidf(self):
        file_path = "test_vocab_similarity_data.csv"

        # Read in test data
        data = pd.read_csv(file_path, sep=",", quotechar='"', na_values="",low_memory=False)
        data.var_desc_1 = data.var_desc_1.fillna("")

        corpora_data = self.partition_by_column(data, column)

        id_col = "varDocID_1"


        doc_col = ["var_desc_1"]
        corpus_builder = CorpusBuilder(doc_col)
        corpus_builder.build_corpora(corpora_data, id_col)

        corpus_builder.calc_tfidf()

        #assert corpus_builder.tfidf_matrix.shape == (nrows, 79)  # using
        #assert corpus_builder.tfidf_matrix[0, 75].round(10) == 0.3692482208


