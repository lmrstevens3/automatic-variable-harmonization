from unittest import TestCase

from automatic_variable_mapping.vocab_similarity import VariableSimilarityCalculator
import numpy as np
import scipy.sparse as sp
import pandas as pd

class TestVariableSimilarityCalculator(TestCase):
    def test_calculate_similarity(self):
        m = np.array([[0.  , 0.  , 0.29, 0.  , 0.  , 0.23, 0.29, 0.  , 0.  , 0.  , 0.29,
                       0.29, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.19, 0.23, 0.  , 0.  ,
                       0.  , 0.29, 0.59, 0.29, 0.  ],
                      [0.  , 0.  , 0.  , 0.31, 0.  , 0.  , 0.  , 0.  , 0.39, 0.  , 0.  ,
                       0.  , 0.39, 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.31, 0.39, 0.39,
                       0.39, 0.  , 0.  , 0.  , 0.  ],
                      [0.36, 0.36, 0.  , 0.29, 0.  , 0.29, 0.  , 0.36, 0.  , 0.36, 0.  ,
                       0.  , 0.  , 0.  , 0.  , 0.36, 0.  , 0.  , 0.23, 0.  , 0.  , 0.  ,
                       0.  , 0.  , 0.  , 0.  , 0.36],
                      [0.  , 0.  , 0.  , 0.  , 0.45, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                       0.  , 0.  , 0.45, 0.45, 0.  , 0.45, 0.45, 0.  , 0.  , 0.  , 0.  ,
                       0.  , 0.  , 0.  , 0.  , 0.  ]])
        result = VariableSimilarityCalculator.calculate_similarity(m, 0)
        expected = pd.DataFrame({'idx': [1, 2, 3], 'score': [0.1188, 0.1104, 0.0000]})

        assert (result['idx'].values == expected['idx'].values).all()
        assert (result['score'].values == expected['score'].values).all()

    def test_init_cache(self):
        self.fail()

    def test_append_cache(self):
        self.fail()

    def test_finalize_cached_output(self):
        self.fail()

    def test_cache_sim_scores(self):
        self.fail()

    def test_score_variables(self):
        self.fail()

    def test_variable_similarity(self):
        self.fail()

    def test_filter_scores(self):
        v = VariableSimilarityCalculator(['x'])
        result = v.filter_scores([('x', 10)], 'x')
        assert result == [('x', 10)]


class Test(TestCase):
    def test_merge_score_results(self):
        self.fail()

    def test_partition(self):
        self.fail()

    def test_vals_differ_in_col(self):
        self.fail()

    def test_vals_differ_in_all_cols(self):
        self.fail()

    def test_val_in_any_row_for_col(self):
        self.fail()

    def test_select_top_sims(self):
        self.fail()

    def test_select_top_sims_by_group(self):
        self.fail()
