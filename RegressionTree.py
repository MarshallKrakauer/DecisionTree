"""Fits on a pandas Dataframe and can predict value on a single row of data."""

import random
from math import floor
from sklearn.metrics import mean_squared_error
from AbstractDecisionTree import print_breadth_first, DecisionTree, get_dataframe
import pandas as pd
import datetime as dt

class RegressionTree(DecisionTree):

    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf'), bootstrap=True, gamma=None):

        super().__init__(dataframe, y_col, parent, depth, random_seed, max_depth,
                         min_sample_split, min_impurity_decrease, bootstrap, gamma)

    def __str__(self):
        """
        Creates string value to represent node

        :return str: String with information about node
        """
        depth_str = 'Depth: ' + str(self.depth)

        # Internal or leaf node
        if self.is_terminal:
            terminal_str = 'Leaf, '
        else:
            terminal_str = 'Int, '

        # Different outputs for internal or leaf node
        if self.split_criterion is not None:
            gini_str = 'MSE: ' + str(round(self.split_criterion, 3))
            split_str = ' at ' + str(round(self.best_split, 3))
            size_str = 'Size: ' + str(len(self.df))
            col_str = 'Feature: ' + self.best_column
            return terminal_str + size_str + ' ' + depth_str + ' ' + gini_str + ' ' + col_str + split_str
        else:
            size_str = 'Size: ' + str(len(self.df))
            prob_str = 'Prob: ' + str(round(self.target_mean, 3))
            return terminal_str + size_str + ' ' + depth_str + ' ' + prob_str

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
        mean_1 = df_1[self.y_col].mean()

        # Calculate mse for each dataframe
        if len_0 > 0:
            mse_0 = mean_squared_error(df_0[self.y_col], [mean_0] * len_0)
        else:
            mse_0 = 0
        if len_1 > 0:
            mse_1 = mean_squared_error(df_1[self.y_col], [mean_1] * len_1)
        else:
            mse_1 = 0

        score = (mse_0 * (len_0 / total_len)) + (mse_1 * (len_1 / total_len))
        return score

    @property
    def target_mean(self):
        return self.df[self.y_col].mean()

    def probability(self):
        raise NotImplementedError("'RegressionTree' object has no attribute 'probability'")

    def predict_proba(self):
        raise NotImplementedError("'RegressionTree' object has no attribute 'predict_proba'")

    def predict(self, data_row):
        if self.is_terminal:
            return self.target_mean
        else:
            value = data_row[self.best_column]
            if value > self.best_split:
                return self.left_child.predict(data_row)
            else:
                return self.right_child.predict(data_row)

    def select_columns(self):
        """Choose subset of columns of which to make splits

        :return list: list of columns which to check for splits
        """
        random.seed(self.random_seed)

        features = [col for col in self.df.columns if col != self.y_col]
        num_columns = floor(len(features) / 3)
        col_list = random.sample(features,num_columns)
        return col_list

    def make_split(self):
        """
        Make the two child nodes based on the given split
        """

        self.find_best_split()

        # If node is terminal, no split is made
        if self.is_terminal:
            return

        self.left_child = RegressionTree(dataframe=self.df[self.df[self.best_column] > self.best_split],
                                         y_col=self.y_col,
                                         parent=self,
                                         depth=self.depth + 1,
                                         max_depth=self.max_depth,
                                         min_sample_split=self.min_sample_split,
                                         min_impurity_decrease=self.min_impurity_decrease,
                                         random_seed=random.random())

        self.right_child = RegressionTree(dataframe=self.df[self.df[self.best_column] <= self.best_split],
                                          y_col=self.y_col,
                                          parent=self,
                                          depth=self.depth + 1,
                                          max_depth=self.max_depth,
                                          min_sample_split=self.min_sample_split,
                                          min_impurity_decrease=self.min_impurity_decrease,
                                          random_seed=random.random())

if __name__ == '__main__':
    _, individual_val, true_value = get_dataframe(False)
    # Overwriting dataframe to get a testing categorical variable
    df = pd.read_csv('testing_categorical.csv')
    dn = RegressionTree(df, 'y')
    # dn.create_tree()
    # print_breadth_first(dn)
    # pred_1 = dn.predict(individual_val)
    # print(pred_1, true_value)
