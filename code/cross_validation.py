import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .data_processing import PreProcessing


class CrossValTraining(PreProcessing):

    def __init__(self, config):
        super().__init__(config)

        self.m_seed = config["seed"]
        self.m_n_folds = config["n_folds"]
        self.m_cv_metric = config["cv_metric"]

        # define crossval kfold instance
        self.m_kf = KFold(n_splits=self.m_n_folds, random_state=self.m_seed, shuffle=True)

        self.m_cv_scores = None
        self.m_cv_predictions = None
        self.m_cv_targets = None

    def cross_validate(self):
        print(f"Commencing {self.m_n_folds}-fold cross-validation ...")
        self.m_cv_scores = []
        self.m_cv_predictions = []
        self.m_cv_targets = []

        for train_index, validation_index in self.m_kf.split(self.m_X):
            X_train, X_val = self.m_X.loc[train_index], self.m_X.loc[validation_index]
            y_train, y_val = self.m_y.loc[train_index], self.m_y.loc[validation_index]

            self.standardize(X_train, X_val)

            self.train(X_train.values, y_train.values)

            y_val_pred = self.predict(X_val.values)
            score = self.score(y_val.values, y_val_pred)

            self.m_cv_scores.append(score)
            self.m_cv_predictions.append(y_val_pred)
            self.m_cv_targets.append(y_val.values)

        print("Cross Validation Concluded. Avg CV score: \n ", np.mean(self.m_cv_scores),
              "std: ", np.std(self.m_cv_scores))
        print(f"Margin of standard variation: {np.std(self.m_cv_scores)/np.mean(self.m_cv_scores)*100} %")

    def score(self, target, prediction):
        # classification tasks
        if self.m_cv_metric == "accuracy":
            score = accuracy_score(target, prediction)
        elif self.m_cv_metric == "f1":
            score = f1_score(target, prediction)
        elif self.m_cv_metric == "precision":
            score = precision_score(target, prediction)
        elif self.m_cv_metric == "recall":
            score = recall_score(target, prediction)
        elif self.m_cv_metric == "roc_auc":
            score = roc_auc_score(target, prediction)
        # regression tasks
        elif self.m_cv_metric == "r2":
            score = r2_score(target, prediction)
        elif self.m_cv_metric == "mean_absolute_error":
            score = mean_absolute_error(target, prediction)
        elif self.m_cv_metric == "mean_squared_error":
            score = mean_squared_error(target, prediction)
        elif self.m_cv_metric == "root_mean_squared_error":
            score = np.sqrt(mean_squared_error(target, prediction))
        elif self.m_cv_metric == "root_mean_squared_log_error":
            score = np.mean(np.sqrt((np.log10(target) - np.log10(prediction))**2))
        else:
            raise NotImplementedError("The given error metric has not been implemented yet in 'code.cross_validation'"
                                      "Please choose one of the following metrics:"
                                      "'accuracy', 'f1', 'precision', 'recall', 'roc_auc'"
                                      "'r2', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error'"
                                      "'root_mean_squared_log_error'")

        return score

    def train(self, X, y):
        raise NotImplementedError("'train' method has not yet been implemented!")

    def predict(self, X):
        raise NotImplementedError("'predict' method has not yet been implemented!")
