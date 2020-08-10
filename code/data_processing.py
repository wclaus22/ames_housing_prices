import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


class PreProcessing:

    def __init__(self, config):
        print("Loading Data ...")

        # get config info
        path = config["path"]

        # get data
        self.m_data = pd.read_csv(f"{path}/train.csv")
        self.m_df_cat = self.m_data.select_dtypes(include="object").copy()
        self.m_df_scalar = self.m_data.select_dtypes(exclude="object").copy().drop(columns=["Id", "SalePrice"])

        # get scalar column names
        self.m_noncat_columns = self.m_df_scalar.columns

        # get y target data
        self.m_y = self.m_data["SalePrice"].copy()

        # define OneHotEncoder
        self.m_encoder = OneHotEncoder(handle_unknown="ignore")

        # finally get X data
        self.m_X = self.impute_and_encode(self.m_data)

        print("Data successfully loaded.")

    def impute_and_encode(self, dataframe, fit=True):

        df_cat = dataframe.select_dtypes(include="object").copy()
        if "SalePrice" in dataframe.columns:
            df_scalar = dataframe.select_dtypes(exclude="object").copy().drop(columns=["Id", "SalePrice"])
        else:
            df_scalar = dataframe.select_dtypes(exclude="object").copy().drop(columns="Id")

        # encode categorical features and add to the scalar features
        if fit:
            X_df = pd.concat([df_scalar, self.m_encoder.fit_transform(df_cat)], axis=1)
        else:
            X_df = pd.concat([df_scalar, self.m_encoder.transform(df_cat)], axis=1)

        # median imputation of NaN values in the scalar features
        bad_columns = X_df.columns[np.where(X_df.isnull().sum() != 0)[0]]
        medians = dataframe[bad_columns].median()
        imputation_values = {column: imputation_val for column, imputation_val in zip(bad_columns, medians)}
        X_df.fillna(value=imputation_values, inplace=True)

        # choose outlier indices from the data exploration notebook
        outlier_indices = [523, 1298, 185, 635]

        return X_df.drop(index=outlier_indices)

    def standardize(self, X_train, X_val):
        # define scaler
        scaler = StandardScaler()

        # define scalar data subsets
        X_train[self.m_noncat_columns] = scaler.fit_transform(X_train[self.m_noncat_columns])
        X_val[self.m_noncat_columns] = scaler.transform(X_val[self.m_noncat_columns])