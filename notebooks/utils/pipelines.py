"""
Module containing all data preprocessing sub-pipelines depending on feature type

Pipelines by sklearn often do not allow the usage of pandas DataFrames
Writing your own pipeline objects by inheriting BaseEstimator and TransfromerMixin
presents a great opportunity to transform data in the form of pandas DataFrames
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def numerical_value_for_rating(rating):
    if rating == "Ex":
        num_value = 5
    elif rating == "Gd":
        num_value = 4
    elif rating == "TA":
        num_value = 3
    elif rating == "FA":
        num_value = 2
    elif rating == "Po":
        num_value = 1
    else:
        num_value = 0
    return num_value


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, selection):
        self.selection = selection

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.selection].copy()


class DataFrameImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy):
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy == "median":
            self.values = {col: X[col].median() for col in X}
        elif self.strategy == "mean":
            self.values = {col: X[col].mean() for col in X}
        else:
            raise ValueError(
                "Invalid replacement strategy given in DataFrameImputer.")
        return self

    def transform(self, X, y=None):
        # choose all columns with NaN values
        for col in X.columns[X.isnull().sum() > 0]:
            X.loc[X[col].isnull(), col] = self.values[col]
        return X


class LogTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in X:
            X[col] = np.log(X[col])
        return X


class ZeroImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.fillna(inplace=False, value=0.)


class SemiContinuousFeatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, log_scale=True):
        self.log_scale = log_scale
        self.feature_indices = {}
        self.feature_medians = {}

    def fit(self, X, y=None):
        for col in X:
            indices = X[col] != 0
            self.feature_indices[col] = indices

            if self.log_scale:
                self.feature_medians[col] = np.median(np.log(X[col][indices]))
            else:
                self.feature_medians[col] = np.median(X[col][indices])

        return self

    def transform(self, X, y=None):
        for col in X:

            if self.log_scale:
                X[col] = [np.log(value) if value !=
                          0. else self.feature_medians[col] for value in X[col]]

            else:
                X[col] = [value if value != 0. else self.feature_medians[col]
                          for value in X[col]]

        return X


class CategoricalNaNImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.all_categories = {col: X[col].unique() for col in X}
        self.most_frequent_categories = {
            col: X[col].mode().values[0] for col in X}
        return self

    def transform(self, X, y=None):
        X.fillna(inplace=True, value="NaN")

        for col in X:
            new_categories = set(X[col].unique()) - \
                set(self.all_categories[col])
            if len(new_categories):  # if there is a category which was not present in training set
                for category in new_categories:
                    # and replace that unknown category with the most frequent!
                    X.loc[X[col] == category,
                          col] = self.most_frequent_categories[col]

        # such that we will finally impute NaN values with medians later
        return X


class SalesClassRanking(BaseEstimator, TransformerMixin):

    def __init__(self, logprice):
        self.logprice = logprice
        self.rankings = {}

    def fit(self, X, y=None):
        X = X.copy()
        X["LogPrice"] = self.logprice

        for col in X.drop(columns="LogPrice"):
            ranking = []
            for category in X[col].unique():
                # get indiv. category LogPrice median
                median_category = X[X[col] == category]["LogPrice"].median()
                ranking.append((median_category, category))

            # sort by LogPrice median
            indices = np.argsort([item[0] for item in ranking])
            order = np.asarray([item[1] for item in ranking])[indices]
            self.rankings[col] = order

        return self

    def transform(self, X, y=None):
        for item in self.rankings:
            X[item] = [self.rankings[item].tolist().index(category)
                       for category in X[item]]

        return X


class OrdinalRanking(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        for col in X:
            X[col] = [numerical_value_for_rating(
                category) for category in X[col]]

        return X


class BinaryFeatureSplitter(BaseEstimator, TransformerMixin):

    def __init__(self, splits):
        self.splits = splits
        assert len(
            self.splits) > 0, "Invalid input to BinaryFeatureSplitter, requires list with splitting values."

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col, split in zip(X.columns, self.splits):
            X[col] = [1 if value > split else 0 for value in X[col]]
        return X


class BinaryFeatureSetter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.all_bin_categories = {col: X[col].unique() for col in X}
        return self

    def transform(self, X, y=None):
        for col in X:
            X[col] = [1 if value == self.all_bin_categories[col]
                      [0] else 0 for value in X[col]]
        return X
