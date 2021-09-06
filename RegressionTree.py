"""Regression Tree. Currently nonfunctional and in testing phase"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error
from HelperFunctions import print_breadth_first

from ClassificationTree import ClassificationTree

data_bunch = load_breast_cancer()
cols = [c.replace(' ', '_') for c in data_bunch['feature_names']]
df = pd.DataFrame(data_bunch['data'], columns=cols)
df['y'] = data_bunch['target']

individual_val = df.loc[0, df.columns != 'y']


class RegressionTree(ClassificationTree):

    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf')):

        super().__init__(dataframe, y_col, parent, depth, random_seed, max_depth,
                     min_sample_split, min_impurity_decrease)

    def calculate_split_criterion(self, column, threshold):
        """
        Calculate the gini impurity for a given split at a given column

        :param column: Column from which to make check split on
        :param threshold: number at which to split
        :return: negative weighted mean squared error term
        """

        # Split into two dataframes
        df_0 = self.df[self.df[column] > threshold]
        len_0 = len(df_0)
        df_1 = self.df[self.df[column] <= threshold]
        len_1 = len(df_1)
        total_len = len_0 + len_1

        if len_0 < self.min_sample_split or len_1 < self.min_sample_split:
            return float('inf')

        mean_0 = df_0[self.y_col].mean()
        mean_1 = df_0[self.y_col].mean()

        print(len_0, len_1)
        # Calculate mse for each dataframe
        if len_0 > 0:
            mse_0 = mean_squared_error(df_0[self.y_col], [mean_0] * len_0)
        else:
            mse_0 = 0
        if len_1 > 0:
            mse_1 = mean_squared_error(df_1[self.y_col], [mean_1] * len_1)
        else:
            mse_1 = 0

        return (mse_0 * len_0 / total_len) + (mse_1 * len_1 / total_len)

    @property
    def target_mean(self):
        return self.df[self.y_col].mean()

    def probability(self):
        raise AttributeError("'RegressionTree' object has no attribute 'probability'")

    def predict_proba(self):
        raise AttributeError("'RegressionTree' object has no attribute 'predict_proba'")

    def predict(self, data_row):
        if self.is_terminal:
            return self.target_mean
        else:
            value = data_row[self.best_column]
            if value > self.best_split:
                return self.left_child.predict(data_row)
            else:
                return self.right_child.predict(data_row)

if __name__ == '__main__':
    dn = RegressionTree(df,'y')
    dn.create_tree()
    print_breadth_first(dn)
    pred_1 = dn.predict(individual_val)
    print(pred_1)