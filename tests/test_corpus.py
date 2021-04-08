import os
from unittest import TestCase

from sklearn.feature_extraction.text import TfidfVectorizer

from automatic_variable_mapping import corpus

import pandas as pd
import numpy as np
import multiprocessing

from automatic_variable_mapping.vocab_similarity import partition


class TestCorpusBuilder(TestCase):
    def test_lemmatize_variable_documentation(self):
        result = corpus.lemmatize_variable_documentation(["hey there little lady"])
        assert result == ["hey", "little", "lady"]

    def test_calc_tfidf(self):
        print(os.getcwd())
        file_path = "tests/test_vocab_similarity_data.csv"

        # Read in test data and sample for test
        data = pd.read_csv(file_path, sep=",",
                           quotechar='"',
                           na_values="",
                           low_memory=False)
        data.var_desc_1 = data.var_desc_1.fillna("")

        id_col = "varDocID_1"

        nrows = 10
        data = [data.sample(nrows, random_state=22895)]

        doc_col = ["var_desc_1"]
        corpora = corpus.build_corpora(doc_col, data, id_col)
        tfidf_matrix = corpus.calc_tfidf(corpora)

        vocabulary = list(set([tok for c in corpora
                               for _, doc in c
                               for tok in doc]))
        check_tf = TfidfVectorizer(tokenizer=lambda x: x,
                                   preprocessor=lambda x: x,
                                   use_idf=True,
                                   norm="l2",
                                   lowercase=False,
                                   vocabulary=vocabulary)
        check_tfidf = check_tf.fit_transform([doc for _, doc in corpora[0]])

        # check CorporaTfidfVectorizer and TfidfVectorizer return same results
        # if corpora = 1 corpus
        assert tfidf_matrix.shape == (nrows, 79)
        # assert check_tf.vocabulary_ == corpus_builder.tf.vocabulary_
        assert tfidf_matrix.shape == check_tfidf.shape
        assert tfidf_matrix.nnz == check_tfidf.nnz
        assert tfidf_matrix.dtype == check_tfidf.dtype
        assert tfidf_matrix[0, 75].round(10) == 0.3692482208
        assert (tfidf_matrix.nnz != check_tfidf.nnz) == 0

    def test_calc_tfidf_multi_corpus(self):
        file_path = "tests/test_vocab_similarity_data.csv"
        # Read in test data
        data = pd.read_csv(file_path, sep=",",
                           quotechar='"',
                           na_values="",
                           low_memory=False)
        data.var_desc_1 = data.var_desc_1.fillna("")

        corpora_data = partition(data, 'study_1')

        # id_col = "varDocID_1"
        # doc_col = ["var_desc_1"]
        # n_corpora = len(corpora_data)
        # corpora = corpus.build_corpora(doc_col, corpora_data, id_col)
        # tfidf_matrix = corpus.calc_tfidf(corpora)
        # tst_words = ['number', 'ecg', 'nutrient', 'blood', 'left', 'heart']

        # def calc_tfidf_single_corpus(corpus_builder_all, corpus_in_corpora):
        #     single_corpus_tfidf = CorpusBuilder(corpus_builder_all.doc_col)
        #     single_corpus_tfidf.corpora = [corpus_in_corpora]
        #     single_corpus_tfidf.calc_tfidf(vocabulary=corpus_builder_all.tf.vocabulary)
        #     return single_corpus_tfidf

        # word_idxs = [tfidf_all.tf.vocabulary_[w] for w in tst_words]
        # corpora_tfidfs = [calc_tfidf_single_corpus(tfidf_all, corpus) for corpus in tfidf_all.corpora]
        # dfs = {word: [1 / (np.exp(tfidf.tf.idf_[tfidf.tf.vocabulary_[word]] - 1))
        #               for tfidf in corpora_tfidfs] for word in tst_words}
        # # check indices for words in tfidf match
        # assert set([tfidf.tf.vocabulary_[w] for w in tst_words for tfidf in corpora_tfidfs]) == set(word_idxs)
        # # check num words represented in each tfidf are equal
        # assert sum([tfidf_all.tfidf_matrix.shape[1] == tfidf.tfidf_matrix.shape[1] for tfidf in corpora_tfidfs]) == 4
        # # check idf calculations are correct result
        # assert all([(np.log(n_corpora / sum(dfs[w])) + 1) for w in tst_words] == tfidf_all.tf.idf_[word_idxs])
        # assert [round(f, 13) for f in dfs['blood']] == [round(f, 13) for f in
        #                                                 [9.0 / 502, 1.0 / 207, 4.0 / 160, 3.0 / 135]]
        # assert [round(f, 13) for f in dfs['nutrient']] == [round(f, 13) for f in
        #                                                    [12.0 / 502, 2.0 / 207, 8.0 / 160, 1.0 / 135]]
        # assert [round(f, 13) for f in dfs['left']] == [round(f, 13) for f in
        #                                                [16.0 / 502, 55.0 / 207, 14.0 / 160, 3.0 / 135]]
        # assert [round(f, 13) for f in dfs['number']] == [round(f, 13) for f in
        #                                                  [16.0 / 502, 43.0 / 207, 5.0 / 160, 12.0 / 135]]

        corpora = [[('race', ['race', 'variable', 'describing',
                              'group', 'individual', 'certain',
                              'characteristic', 'common', 'owing',
                              'common', 'inheritance']),
                    ('gender', ['gender', 'variable', 'describing',
                                'self', 'identified', 'category',
                                'basis', 'sex']),
                    ('sex', ['sex', 'variable', 'descriptive',
                             'biological', 'characterization', 'based',
                             'gamete', 'gonad', 'individual']),
                    ('ethnicity', ['affiliation', 'due', 'shared',
                                   'cultural', 'background'])]]
        tfidf_matrix = corpus.calc_tfidf(corpora)
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

        assert (np.around(tfidf_matrix.todense(), decimals=2) == m).all()

    def test_build_corpora(self):
        data = pd.DataFrame({
            'documentation': [
                'race is a variable describing a group of individuals having certain characteristics in common, '
                'owing to a common inheritance.',
                'gender is a variable describing the self-identified category on the basis of sex.',
                'sex as a variable is descriptive of the biological characterization based on the gametes or gonads '
                'of an individual.',
                'The affiliation due to shared cultural background.'],
            'alt documentation': ['1', '2', '1 2', '2 1'],
            'id': ['race', 'gender', 'sex', 'ethnicity']
        })
        corpora = corpus.build_corpora(['documentation'], [data], 'id')

        assert corpora == [[('race', ['race', 'variable', 'describing',
                                      'group', 'individual', 'certain',
                                      'characteristic', 'common', 'owing',
                                      'common', 'inheritance']),
                            ('gender', ['gender', 'variable', 'describing',
                                        'self', 'identified', 'category',
                                        'basis', 'sex']),
                            ('sex', ['sex', 'variable', 'descriptive',
                                     'biological', 'characterization', 'based',
                                     'gamete', 'gonad', 'individual']),
                            ('ethnicity', ['affiliation', 'due', 'shared',
                                           'cultural', 'background'])]]

    def test_all_docs(self):
        corpora = [[('race', ['race', 'variable', 'describing',
                              'group', 'individual', 'certain',
                              'characteristic', 'common', 'owing',
                              'common', 'inheritance']),
                    ('gender', ['gender', 'variable', 'describing',
                                'self', 'identified',
                                'category', 'basis', 'sex']),
                    ('sex', ['sex', 'variable', 'descriptive',
                             'biological', 'characterization', 'based',
                             'gamete', 'gonad', 'individual']),
                    ('ethnicity', ['affiliation', 'due', 'shared',
                                   'cultural', 'background'])]]

        result = corpus.all_docs(corpora)
        assert result == [['race', 'variable', 'describing',
                           'group', 'individual', 'certain',
                           'characteristic', 'common', 'owing',
                           'common', 'inheritance'],
                          ['gender', 'variable', 'describing',
                           'self', 'identified', 'category',
                           'basis', 'sex'],
                          ['sex', 'variable', 'descriptive',
                           'biological', 'characterization', 'based',
                           'gamete', 'gonad', 'individual'],
                          ['affiliation', 'due', 'shared',
                           'cultural', 'background']]

    def test_build_corpus(self):
        data = pd.DataFrame({
            'documentation': [
                'race is a variable describing a group of individuals having certain characteristics in common, '
                'owing to a common inheritance.',
                'gender is a variable describing the self-identified category on the basis of sex.',
                'sex as a variable is descriptive of the biological characterization based on the gametes or gonads '
                'of an individual.',
                'The affiliation due to shared cultural background.'],
            'alt documentation': ['1', '2', '1 2', '2 1'],
            'id': ['race', 'gender', 'sex', 'ethnicity']
        })

        doc_col = ['documentation']

        result = corpus.build_corpus(doc_col, data, 'id',
                                     num_cpus=multiprocessing.cpu_count())

        assert result == [('race', ['race', 'variable', 'describing',
                                    'group', 'individual', 'certain',
                                    'characteristic', 'common', 'owing',
                                    'common', 'inheritance']),
                          ('gender', ['gender', 'variable', 'describing',
                                      'self', 'identified', 'category',
                                      'basis', 'sex']),
                          ('sex', ['sex', 'variable', 'descriptive',
                                   'biological', 'characterization', 'based',
                                   'gamete', 'gonad', 'individual']),
                          ('ethnicity', ['affiliation', 'due', 'shared',
                                         'cultural', 'background'])]
