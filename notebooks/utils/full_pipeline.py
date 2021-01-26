"""
This module provides the full data pipeline

A detailed analysis is found in the feature engineering notebook
"""
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

from utils.pipelines import *

full_path = "~/Projects/ames_housing_prices/data/train.csv"

data = pd.read_csv(full_path)
logprice = np.log(data["SalePrice"])

########################################################################
# Step I: Define the features
########################################################################

all_features = []

# get continuous features log-scaled features
continuous_sel = ["LotArea", "1stFlrSF", "GrLivArea"]
all_features += continuous_sel

# log scaled non-zero values + imputation with median
# removed TotalBsmtSF (multicollinearity)
semicon_scaled_sel = ['LotFrontage', 'WoodDeckSF']
all_features += semicon_scaled_sel

# non scaled non-zero values + imputation with median
semicon_non_sel = ['BsmtFinSF1', 'GarageArea', "2ndFlrSF"]
all_features += semicon_non_sel

# true integer valued features (e.g. num baths, num rooms, OverallQual)
chosen_discrete_num_features = [
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr',  # removed TotRmsAbvGrd & Fireplaces (multicollinearity)
    'GarageCars', "OverallQual", "OverallCond"]
all_features += chosen_discrete_num_features

# rankable features based on median SalePrice value (e.g. Neighborhood)
chosen_ordinal_ranked_features = [
    "MSSubClass", "MSZoning", "Neighborhood", "LotShape", "LandContour", "MasVnrType", "GarageType", "GarageFinish",
    # removed Exterior2nd (multicollinearity)
    "BsmtExposure", "Exterior1st", "LotConfig", "Foundation"
]
all_features += chosen_ordinal_ranked_features

# already ranked features by category (e.g. KitchenQual)
ordinal_features = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                    "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual"]  # removed PoolQC & GarageCond (multicollinearity)
all_features += ordinal_features

# semicontinuous and yearly timestamp features turned into binary split features
binary_categorical_features = ["YearBuilt", "YearRemodAdd",  "PoolArea",
                               "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
all_features += binary_categorical_features

# binary categorical features
custom_features = ["Street", "CentralAir"]
all_features += custom_features

########################################################################
# Step II: Define the feature extraction and engineering sub-pipelines
########################################################################

continuous_log_features = Pipeline([
    ("Selector", DataFrameSelector(continuous_sel)),
    ("LogTransform", LogTransformer()),
    ("MedianImputer", DataFrameImputer(strategy="median")),
    ("StandardScaler", StandardScaler())
])

semicon_log_features = Pipeline([
    ("Selector", DataFrameSelector(semicon_scaled_sel)),
    ("ZeroImputer", ZeroImputer()),
    ("SemiConLogFeatureTransform", SemiContinuousFeatureTransform(log_scale=True)),
    ("StandardScaler", StandardScaler())
])

semicon_nonscaled_features = Pipeline([
    ("Selector", DataFrameSelector(semicon_non_sel)),
    ("ZeroImputer", ZeroImputer()),
    ("SemiConLogFeatureTransform", SemiContinuousFeatureTransform(log_scale=False)),
    ("StandardScaler", StandardScaler())
])

discrete_features = Pipeline([
    ("Selector", DataFrameSelector(chosen_discrete_num_features)),
    ("Imputer", SimpleImputer(strategy="most_frequent")),
    ("StandardScaler", StandardScaler())
])

ordinal_ranked_features = Pipeline([
    ("Selector", DataFrameSelector(
        chosen_ordinal_ranked_features)),
    ("Imputer", CategoricalNaNImputer()),
    ("Ranking", SalesClassRanking(logprice)),
    ("StandardScaler", StandardScaler())
])

ordinal_features = Pipeline([
    ("Selector", DataFrameSelector(ordinal_features)),
    ("Ranking", OrdinalRanking()),
    ("Imputer", SimpleImputer(strategy="median")),
    ("StandardScaler", StandardScaler())
])

binary_feature_extractor = Pipeline([
    ("Selector", DataFrameSelector(binary_categorical_features)),
    ("BinarySplitter", BinaryFeatureSplitter(
        splits=[1970, 1984, 0, 0, 0, 0, 0])),
    ("Imputer", SimpleImputer(strategy="most_frequent"))
])

custom_bin_feature_extractor = Pipeline([
    ("Selector", DataFrameSelector(custom_features)),
    ("BinaryFeatureSetter", BinaryFeatureSetter()),
    ("Imputer", SimpleImputer(strategy="most_frequent"))
])

########################################################################
# Step III: Combine all of the sub-pipelines into the full pipeline
########################################################################

transformer_list = [
    ("continuous_log_features", continuous_log_features),
    ("semicon_log_features", semicon_log_features),
    ("semicon_nonscaled_features", semicon_nonscaled_features),
    ("discrete_features", discrete_features),
    ("ordinal_ranked_features", ordinal_ranked_features),
    ("ordinal_features", ordinal_features),
    ("binary_feature_extractor", binary_feature_extractor),
    ("custom_bin_feature_extractor", custom_bin_feature_extractor)
]


class FullPipeline(FeatureUnion):
    def __init__(self,
                 transformer_list=transformer_list,
                 all_features=all_features):

        self.all_features = all_features
        self.transformer_list = transformer_list
        super().__init__(transformer_list=transformer_list)
