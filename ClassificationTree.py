"""Fits on a pandas Dataframe and can predict the class and probability on a single row of data."""

from math import sqrt, floor
import random
from AbstractDecisionTree import print_breadth_first, DecisionTree, get_dataframe

class ClassificationTree(DecisionTree):
    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf')):

        super().__init__(dataframe, y_col, parent, depth, random_seed, max_depth,
                     min_sample_split, min_impurity_decrease)

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
            gini_str = 'Gini: ' + str(round(self.split_criterion, 3))
            split_str = ' at ' + str(round(self.best_split, 3))
            size_str = 'Size: ' + str(len(self.df))
            col_str = 'Feature: ' + self.best_column
            return terminal_str + size_str + ' ' + depth_str + ' ' +  gini_str + ' ' + col_str +  split_str
        else:
            size_str = 'Size: ' + str(len(self.df))
            prob_str = 'Prob: ' + str(round(self.probability,3))
            return terminal_str + size_str + ' ' + depth_str + ' ' + prob_str

    @property
    def probability(self):
        return self.df[self.y_col].mean()

    def calculate_split_criterion(self, column, threshold):
        """
        Calculate the gini impurity for a given split at a given column

        :param column: Column from which to make check split on
        :param threshold: number at which to split
        :return: gini score
        """
        gini = 0  # returning value

        # Split into two dataframes
        df_0 = self.df[self.df[column]  > threshold]
        len_0 = len(df_0)
        df_1 = self.df[self.df[column] <= threshold]
        len_1 = len(df_1)

        if len_0 < self.min_sample_split or len_1 < self.min_sample_split:
            return float('inf')
        # Calculate gini score for each dataframe
        for split in [df_0, df_1]:
            temp_score = 0
            if len(split) > 0:

                # Calculate probability score for each class
                probability_0 = len(split[split[self.y_col] == 0]) / len(split)
                probability_1 = len(split[split[self.y_col] == 1]) / len(split)
                temp_score += (probability_0 * probability_0 + probability_1 * probability_1)
            gini += (1 - temp_score) * (len(split) / len(self.df))

        return gini

    def select_columns(self):
        """Choose subset of columns of which to make splits

        :return list: list of columns which to check for splits
        """
        random.seed(self.random_seed)
        num_columns = floor(sqrt(len(self.df.columns)))
        features = [col for col in self.df.columns if col != self.y_col]

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

        self.left_child = ClassificationTree(dataframe=self.df[self.df[self.best_column] > self.best_split],
                                             y_col=self.y_col,
                                             parent=self,
                                             depth=self.depth+1,
                                             max_depth=self.max_depth,
                                             min_sample_split=self.min_sample_split,
                                             min_impurity_decrease=self.min_impurity_decrease,
                                             random_seed=random.random())

        self.right_child = ClassificationTree(dataframe= self.df[self.df[self.best_column] <= self.best_split],
                                              y_col=self.y_col,
                                              parent=self,
                                              depth=self.depth+1,
                                              max_depth=self.max_depth,
                                              min_sample_split=self.min_sample_split,
                                              min_impurity_decrease=self.min_impurity_decrease,
                                              random_seed=random.random())

    def create_tree(self):
        """Creates decision tree from a root node"""
        self.make_split()
        if not self.is_terminal:
            self.left_child.create_tree()
            self.right_child.create_tree()

    def predict_proba(self, data_row):
        if self.is_terminal:
            return self.probability
        else:
            value = data_row[self.best_column]
            if value > self.best_split:
                return self.left_child.predict_proba(data_row)
            else:
                return self.right_child.predict_proba(data_row)

    def predict(self, data_row, cutoff=0.5):
        prediction = self.predict_proba(data_row)

        if prediction < cutoff:
            return 0
        else:
            return 1

if __name__ == '__main__':
    df, individual_val, true_value = get_dataframe(True)
    dn = ClassificationTree(df, 'y', max_depth=4, min_sample_split=5, min_impurity_decrease=0, random_seed=777)
    dn.create_tree()
    print_breadth_first(dn)
    probability_0_ = dn.predict_proba(individual_val)
    probability_1_ = dn.predict(individual_val)
    print(probability_0_, probability_1_, true_value)
