import matrixprofile as mp

class MatrixProfile():
    def __init__(self, window = 100, cross_correlation=False, n_jobs=1):
        self.window = window
        self.cross_correlation = cross_correlation
        self.model_name = 'MatrixProfile'
        self.n_jobs = 1

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.cross_correlation:
            self.profile = mp.compute_mpx(X, windows=self.window, cross_correlation=self.cross_correlation, n_jobs=self.n_jobs)
        else:
            self.profile = mp.compute(X, windows=self.window, n_jobs=self.n_jobs)
        self.decision_scores_ = self.profile['mp']
        return self
    
    def top_k_discords(self, k=5):
        discords = mp.discover.discords(self.profile, exclusion_zone=self.window//2, k=k)
        return discords['discords']


class MatrixProfile_original():
    def __init__(self, window = 100, sample_pct=1, cross_correlation=False):
        self.window = window
        self.sample_pct = sample_pct
        self.cross_correlation = cross_correlation
        self.model_name = 'MatrixProfile'

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.profile = mp.compute(X, windows=self.window, n_jobs=-1)
        self.decision_scores_ = self.profile['mp']
        return self
    
    def top_k_discords(self, k=5):
        discords = mp.discover.discords(self.profile, exclusion_zone=self.window//2, k=k)
        return discords['discords']
