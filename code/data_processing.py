import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


class PreProcessing:

    def __init__(self, config):
        print("Loading Data ...")

        # get config info
        path = config["path"]

        # get data
        self.m_data = pd.read_csv(f"{path}/train.csv").drop(columns="Id")
        self.m_df_cat = self.m_data.select_dtypes(include="object").copy()
        self.m_df_scalar = self.m_data.select_dtypes(exclude="object").copy()

        # get scalar column names
        self.m_noncat_columns = self.m_df_scalar.drop(columns="SalePrice").columns

        # get y target data
        self.m_y = self.m_data["SalePrice"].copy()

        # encode categorical features and add to the scalar features
        X_df = pd.concat([self.m_df_scalar, pd.get_dummies(self.m_df_cat)], axis=1)

        # median imputation of NaN values in the scalar features
        bad_columns = X_df.columns[np.where(X_df.isnull().sum() != 0)[0]]
        medians = self.m_data[bad_columns].median()
        imputation_values = {column: imputation_val for column, imputation_val in zip(bad_columns, medians)}
        X_df.fillna(value=imputation_values, inplace=True)

        # finally get X data
        self.m_X = X_df

        print("Data successfully loaded.")

    def standardize(self, X_train, X_val):
        # define scaler
        scaler = StandardScaler()

        # define scalar data subsets
        X_train[self.m_noncat_columns] = scaler.fit_transform(X_train[self.m_noncat_columns])
        X_val[self.m_noncat_columns] = scaler.transform(X_val[self.m_noncat_columns])