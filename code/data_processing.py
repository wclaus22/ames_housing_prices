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
        self.final_columns = None

        # get y target data
        self.m_y = self.m_data["SalePrice"].copy()

        # define OneHotEncoder
        self.m_encoder = OneHotEncoder(handle_unknown="ignore")

        # finally get X data
        self.m_X = self.impute_and_encode(self.m_data)

        # choose outlier indices from the data exploration notebook
        self.outlier_indices = [523, 1298, 185, 635]

        print("Data successfully loaded.")

    def impute_and_encode(self, dataframe):

        df_cat = dataframe.select_dtypes(include="object").copy()
        if "SalePrice" in dataframe.columns:
            df_scalar = dataframe.select_dtypes(exclude="object").copy().drop(columns=["Id", "SalePrice"])
        else:
            df_scalar = dataframe.select_dtypes(exclude="object").copy().drop(columns="Id")

        # encode categorical features and add to the scalar features
        X_df = pd.concat([df_scalar, pd.get_dummies(df_cat)], axis=1)

        if self.final_columns is None:
            self.final_columns = X_df.columns

        if len(X_df.columns) != len(self.final_columns):
            diff_columns = list(set(self.final_columns) - set(X_df))

            for column in diff_columns:
                # zero imputation
                X_df[column] = np.zeros(len(X_df))

        # median imputation of NaN values in the scalar features
        bad_columns = X_df.columns[np.where(X_df.isnull().sum() != 0)[0]]
        medians = dataframe[bad_columns].median()
        imputation_values = {column: imputation_val for column, imputation_val in zip(bad_columns, medians)}
        X_df.fillna(value=imputation_values, inplace=True)

        return X_df

    def standardize(self, X_train, X_val):
        # define scaler
        scaler = StandardScaler()

        # define scalar data subsets
        X_train[self.m_noncat_columns] = scaler.fit_transform(X_train[self.m_noncat_columns])
        X_val[self.m_noncat_columns] = scaler.transform(X_val[self.m_noncat_columns])