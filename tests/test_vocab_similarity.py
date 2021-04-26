from unittest import TestCase
import numpy as np
import pandas as pd
from automatic_variable_mapping.vocab_similarity import VariableSimilarityCalculator, default_pairable, \
    calculate_similarity, filter_scores, identity, select_top_sims_by_group


class TestVariableSimilarityCalculator(TestCase):

    def test_score_docs(self):
        # corpora = [[('race', ['race', 'variable', 'describing', 'group', 'individual', 'certain',
        #                       'characteristic', 'common', 'owing', 'common', 'inheritance']),
        #             ('gender', ['gender', 'variable', 'describing', 'self', 'identified', 'category', 'basis','sex']),
        #             ('sex', ['sex', 'variable', 'descriptive', 'biological', 'characterization', 'based',
        #                      'gamete', 'gonad', 'individual']),
        #             ('ethnicity', ['affiliation', 'due', 'shared', 'cultural', 'background'])]]
        ref_ids = ['race', 'gender', 'sex']
        doc_ids = ['race', 'gender', 'sex', 'ethnicity']
        m = np.array([[0., 0., 0.29, 0., 0., 0.23, 0.29, 0., 0., 0., 0.29,
                       0.29, 0., 0., 0., 0., 0., 0., 0.19, 0.23, 0., 0.,
                       0., 0.29, 0.59, 0.29, 0.],
                      [0., 0., 0., 0.31, 0., 0., 0., 0., 0.39, 0., 0.,
                       0., 0.39, 0., 0., 0., 0., 0., 0.25, 0.31, 0.39, 0.39,
                       0.39, 0., 0., 0., 0.],
                      [0.36, 0.36, 0., 0.29, 0., 0.29, 0., 0.36, 0., 0.36, 0.,
                       0., 0., 0., 0., 0.36, 0., 0., 0.23, 0., 0., 0.,
                       0., 0., 0., 0., 0.36],
                      [0., 0., 0., 0., 0.45, 0., 0., 0., 0., 0., 0.,
                       0., 0., 0.45, 0.45, 0., 0.45, 0.45, 0., 0., 0., 0.,
                       0., 0., 0., 0., 0.]])

        scores = VariableSimilarityCalculator(ref_ids, pairable=default_pairable)

        result = scores.score_docs(doc_ids, m)
        expected = pd.DataFrame([('race', 'gender', 0.1188),
                                 ('race', 'sex', 0.1104),
                                 ('gender', 'sex', 0.1474),
                                 ('gender', 'race', 0.1188),
                                 ('sex', 'gender', 0.1474),
                                 ('sex', 'race', 0.1104)],
                                columns=['reference var', 'paired var', 'score'])
        assert (result.equals(expected))

    def test_score_docs_out_of_order(self):
        # corpora = [[('race', ['race', 'variable', 'describing', 'group', 'individual', 'certain',
        #                       'characteristic', 'common', 'owing', 'common', 'inheritance']),
        #             ('gender', ['gender', 'variable', 'describing', 'self', 'identified', 'category', 'basis','sex']),
        #             ('sex', ['sex', 'variable', 'descriptive', 'biological', 'characterization', 'based',
        #                      'gamete', 'gonad', 'individual']),
        #             ('ethnicity', ['affiliation', 'due', 'shared', 'cultural', 'background'])]]
        ref_ids = ['sex', 'race', 'gender']
        doc_ids = ['race', 'gender', 'sex', 'ethnicity']
        m = np.array([[0., 0., 0.29, 0., 0., 0.23, 0.29, 0., 0., 0., 0.29,
                       0.29, 0., 0., 0., 0., 0., 0., 0.19, 0.23, 0., 0.,
                       0., 0.29, 0.59, 0.29, 0.],
                      [0., 0., 0., 0.31, 0., 0., 0., 0., 0.39, 0., 0.,
                       0., 0.39, 0., 0., 0., 0., 0., 0.25, 0.31, 0.39, 0.39,
                       0.39, 0., 0., 0., 0.],
                      [0.36, 0.36, 0., 0.29, 0., 0.29, 0., 0.36, 0., 0.36, 0.,
                       0., 0., 0., 0., 0.36, 0., 0., 0.23, 0., 0., 0.,
                       0., 0., 0., 0., 0.36],
                      [0., 0., 0., 0., 0.45, 0., 0., 0., 0., 0., 0.,
                       0., 0., 0.45, 0.45, 0., 0.45, 0.45, 0., 0., 0., 0.,
                       0., 0., 0., 0., 0.]])

        scores = VariableSimilarityCalculator(ref_ids, pairable=default_pairable)

        result = scores.score_docs(doc_ids, m)
        expected = pd.DataFrame([('race', 'gender', 0.1188),
                                 ('race', 'sex', 0.1104),
                                 ('gender', 'sex', 0.1474),
                                 ('gender', 'race', 0.1188),
                                 ('sex', 'gender', 0.1474),
                                 ('sex', 'race', 0.1104)],
                                columns=['reference var', 'paired var', 'score'])
        assert (result.equals(expected))

    def test_calculate_similarity(self):
        m = np.array([[0., 0., 0.29, 0., 0., 0.23, 0.29, 0., 0., 0., 0.29,
                       0.29, 0., 0., 0., 0., 0., 0., 0.19, 0.23, 0., 0.,
                       0., 0.29, 0.59, 0.29, 0.],
                      [0., 0., 0., 0.31, 0., 0., 0., 0., 0.39, 0., 0.,
                       0., 0.39, 0., 0., 0., 0., 0., 0.25, 0.31, 0.39, 0.39,
                       0.39, 0., 0., 0., 0.],
                      [0.36, 0.36, 0., 0.29, 0., 0.29, 0., 0.36, 0., 0.36, 0.,
                       0., 0., 0., 0., 0.36, 0., 0., 0.23, 0., 0., 0.,
                       0., 0., 0., 0., 0.36],
                      [0., 0., 0., 0., 0.45, 0., 0., 0., 0., 0., 0.,
                       0., 0., 0.45, 0.45, 0., 0.45, 0.45, 0., 0., 0., 0.,
                       0., 0., 0., 0., 0.]])
        result = calculate_similarity(m, 0)
        expected = [(1, 0.1188), (2, 0.1104), (3, 0)]

        assert result == expected

    def test_filter_scores(self):
        scores = [(1, 0.1188), (2, 0.1104), (3, 0)]
        result = list(filter_scores([], identity, scores, 'x'))
        assert result == scores

    def test_select_top_sims_by_group(self):
        ref_doc_scores = [(2, 0.983),
                          (4, 0.9621),
                          (8, 0.823),
                          (1, 0.715),
                          (0, 0.623),
                          (5, 0.315),
                          (6, 0.283),
                          (7, 0.015),
                          (3, 0.003)]
        doc_ids = ['pid1', 'pid2', 'pid3', 'pid4', 'pid5', 'pid6', 'pid7', 'pid8', 'pid9']
        id_group_dict = {'pid1': 'g1',
                         'pid2':'g2',
                         'pid3':'g4',
                         'pid5':'g1',
                         'pid6':'g2',
                         'pid7':'g1',
                         'pid8':'g4',
                         'pid9':'g2'}
        result = select_top_sims_by_group(ref_doc_scores, 2, doc_ids, id_group_dict)
        expected = [(2, 0.983), (7, 0.015),
                    (8, 0.823), (1, 0.715),
                    (4, 0.9621), (0, 0.623)]
        assert result == expected