from sklearn.linear_model import Ridge

from .cross_validation import CrossValTraining


class RidgeRegression(CrossValTraining):
    """
    RidgeRegression uses the sklearn Ridge class, check the individual keyword arguments
    of sklearn.linear_model.Ridge to choose the right arguments for the Ridge Regression
    """
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args:
            config : dict
                a dict containing all important information for this model and the preprocessing, as well as crossval
                path : str
                    path to dataset
                seed : int
                    crossval split seed
                n_folds : int
                    number of crossval folds
                cv_metric : str
                    choice of metric for the crossval
        **kwargs (see sklearn.linear_model.Ridge)

        Returns
        -------
        NoneType
            None

        """
        super().__init__(args[0])
        self.m_model = None
        self.kwargs = {**kwargs}

    def train(self, X, y):
        self.m_model = Ridge(**self.kwargs)
        self.m_model.fit(X, y)

    def predict(self, X):
        return self.m_model.predict(X)






