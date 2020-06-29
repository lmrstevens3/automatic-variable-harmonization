# noinspection PyProtectedMember
from sklearn.feature_extraction.text import TfidfVectorizer, _document_frequency
from sklearn.utils.validation import FLOAT_DTYPES, check_array
import scipy.sparse as sp
import numpy as np


def fit_corpora(tfidf, Xs):
    """Learn the idf vector (global term weights)

    Parameters
    ----------
    Xs : list of sparse matricies, X [n_samples, n_features]
        a matrix of term/token counts for each corpus in Xs

    tfidf: TfidfVectorizer._tfidf object
    """
    full_df = 0
    if tfidf.use_idf:
        n_samples, n_features = Xs[0].shape
        dtype = Xs[0].dtype if Xs[0].dtype in FLOAT_DTYPES else np.float64
        for X in Xs:
            X = check_array(X, accept_sparse=('csr', 'csc'))
            if not sp.issparse(X):
                X = sp.csr_matrix(X)

            n_samples, _ = X.shape

            df = _document_frequency(X).astype(dtype)

            # perform idf smoothing if required
            df += int(tfidf.smooth_idf)
            n_samples += int(tfidf.smooth_idf)

            full_df += df / n_samples

        n_corpora = len(Xs)

        # log+1 instead of log makes sure terms with zero idf don't get
        # suppressed entirely.
        idf = np.log(n_corpora / full_df) + 1
        tfidf._idf_diag = sp.diags(idf, offsets=0,
                                   shape=(n_features, n_features),
                                   format='csr',
                                   dtype=dtype)

    return tfidf


class CorporaTfidfVectorizer(TfidfVectorizer):

    def fit(self, corpora, y=None):
        """Learn vocabulary and idf for each corpus in corpora.

        Parameters
        ----------
        corpora : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self : TfidfVectorizer
        :param corpora: corpora list
        :param y:
        """
        self._check_params()
        Xs = [super(TfidfVectorizer, self).fit_transform(raw_documents) for raw_documents in corpora]
        fit_corpora(self._tfidf, Xs)
        return self
