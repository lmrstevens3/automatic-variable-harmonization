from unittest import TestCase

import numpy as np
import scipy.sparse as sp

from automatic_variable_mapping.vocab_similarity import VariableSimilarityCalculator, calculate_similarity, filter_scores, identity


class TestVariableSimilarityCalculator(TestCase):
    def test_calculate_similarity(self):
        m = np.array([[0., 0., 0.29,
                       0., 0., 0.23,
                       0.29, 0., 0.,
                       0., 0.29, 0.29,
                       0., 0., 0.,
                       0., 0., 0.,
                       0.19, 0.23, 0.,
                       0., 0., 0.29,
                       0.59, 0.29, 0.],
                      [0., 0., 0.,
                       0.31, 0., 0.,
                       0., 0., 0.39,
                       0., 0., 0.,
                       0.39, 0., 0.,
                       0., 0., 0.,
                       0.25, 0.31, 0.39,
                       0.39, 0.39, 0.,
                       0., 0., 0.],
                      [0.36, 0.36, 0.,
                       0.29, 0., 0.29,
                       0., 0.36, 0.,
                       0.36, 0., 0.,
                       0., 0., 0.,
                       0.36, 0., 0.,
                       0.23, 0., 0.,
                       0., 0., 0.,
                       0., 0., 0.36],
                      [0., 0., 0.,
                       0., 0.45, 0.,
                       0., 0., 0.,
                       0., 0., 0.,
                       0., 0.45, 0.45,
                       0., 0.45, 0.45,
                       0., 0., 0.,
                       0., 0., 0.,
                       0., 0., 0.]])
        m = sp.csc_matrix(m)
        result = calculate_similarity(m, 0)
        expected = [(1, 0.1188),
                    (2, 0.1104),
                    (3, 0)]

        assert result == expected

    # def test_init_cache(self):
    #     self.fail()

    # def test_append_cache(self):
    #     self.fail()

    # def test_finalize_cached_output(self):
    #     self.fail()

    # def test_cache_sim_scores(self):
    #     self.fail()

    def test_score_variables(self):
        v = VariableSimilarityCalculator(['race'])
        corpora = [
            [('race', ['race', 'variable', 'describing', 'group', 'individual', 'certain', 'characteristic', 'common',
                       'owing', 'common', 'inheritance']),
             ('gender', ['gender', 'variable', 'describing', 'self', 'identified', 'category', 'basis', 'sex']),
             ('sex',
              ['sex', 'variable', 'descriptive', 'biological', 'characterization', 'based', 'gamete', 'gonad',
               'individual']), ('ethnicity', ['affiliation', 'due', 'shared', 'cultural', 'background'])]]
        m = np.array([[0., 0., 0.29,
                       0., 0., 0.23,
                       0.29, 0., 0.,
                       0., 0.29, 0.29,
                       0., 0., 0.,
                       0., 0., 0.,
                       0.19, 0.23, 0.,
                       0., 0., 0.29,
                       0.59, 0.29, 0.],
                      [0., 0., 0.,
                       0.31, 0., 0.,
                       0., 0., 0.39,
                       0., 0., 0.,
                       0.39, 0., 0.,
                       0., 0., 0.,
                       0.25, 0.31, 0.39,
                       0.39, 0.39, 0.,
                       0., 0., 0.],
                      [0.36, 0.36, 0.,
                       0.29, 0., 0.29,
                       0., 0.36, 0.,
                       0.36, 0., 0.,
                       0., 0., 0.,
                       0.36, 0., 0.,
                       0.23, 0., 0.,
                       0., 0., 0.,
                       0., 0., 0.36],
                      [0., 0., 0.,
                       0., 0.45, 0.,
                       0., 0., 0.,
                       0., 0., 0.,
                       0., 0.45, 0.45,
                       0., 0.45, 0.45,
                       0., 0., 0.,
                       0., 0., 0.,
                       0., 0., 0.]])

        m = sp.csr_matrix(m)

        result = v.score_variables(corpora[0], m, num_cpus=2)

        expected = np.array([['race', 'gender', 0.1188],
                             ['race', 'sex', 0.1104],
                             ['race', 'ethnicity', 0.0]],
                            dtype='object')

        print result.values
        print expected
        assert (result.values == expected).all()

    # def test_variable_similarity(self):
    #     # v = VariableSimilarityCalculator(['x'])
    #     # v.variable_similarity()
    #     self.fail()

    def test_filter_scores(self):
        v = VariableSimilarityCalculator(['x'])

        scores = [(1, 0.1188), (2, 0.1104), (3, 0)]

        result = list(filter_scores([], identity, scores, 'x'))

        assert result == scores


# class Test(TestCase):
#     def test_merge_score_results(self):
#         self.fail()

#     def test_partition(self):
#         self.fail()

#     def test_vals_differ_in_col(self):
#         self.fail()

#     def test_vals_differ_in_all_cols(self):
#         self.fail()

#     def test_val_in_any_row_for_col(self):
#         self.fail()

#     def test_select_top_sims(self):
#         self.fail()

#     def test_select_top_sims_by_group(self):
#         self.fail()
