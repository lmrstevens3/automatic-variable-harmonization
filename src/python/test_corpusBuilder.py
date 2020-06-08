import pandas as pd
from unittest import TestCase

from src.python.Vocab_Similarity import CorpusBuilder


class TestCorpusBuilder(TestCase):
    def test_calc_tfidf(self):
        file_path = "test_vocab_similarity_data.csv"

        # READ IN DATA -- 07.17.19
        data = pd.read_csv(file_path, sep=",", quotechar='"', na_values="",
                           low_memory=False)  # when reading in data, check to see if there is "\r" if
        # not then don't use "lineterminator='\n'", otherwise u
        data.units_1 = data.units_1.fillna("")
        data.dbGaP_dataset_label_1 = data.dbGaP_dataset_label_1.fillna("")
        data.var_desc_1 = data.var_desc_1.fillna("")
        data.var_coding_labels_1 = data.var_coding_labels_1.fillna("")

        id_col = "varDocID_1"

        # SCORE DATA + WRITE OUT
        nrows = 10
        data = data.sample(nrows, random_state=22895)

        doc_col = ["var_desc_1"]
        corpus_builder = CorpusBuilder(doc_col)
        corpus_builder.build_corpus(data, id_col)

        corpus_builder.calc_tfidf()

        assert corpus_builder.tfidf_matrix.shape == (nrows, 79)  # using
        assert corpus_builder.tfidf_matrix[0, 75].round(10) == 0.3692482208
