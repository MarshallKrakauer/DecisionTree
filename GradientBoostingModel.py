"""Gradient Boosting Classier Model. Currently supports classification and regression"""

import random
from GBCTree import GBCTree
from GBRTree import GBRTree
from AbstractDecisionTree import print_breadth_first, get_dataframe
import numpy as np

class GBCModel:

    def __init__(self, dataframe, y_col='target', random_seed=0.0, max_depth=3,
                 min_sample_split=0, gamma=1, lambda_=1,
                 previous_similarity=0, eta=0.3, num_trees=3, is_classification=True):
        self.df = dataframe
        self.y_col = y_col
        self.random_seed = random_seed
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.gamma = gamma
        self.lambda_ = lambda_
        self.parent_similarity = previous_similarity
        self.eta = eta
        self.num_trees = num_trees
        self.tree_list = []
        self.is_classification = is_classification

    def make_gbm(self):
        gbm_dataframe = self.df

        for i in range(self.num_trees):
            tree = self.make_tree(i==0, gbm_dataframe)
            tree.create_tree()
            tree.update_probabilities()
            self.tree_list.append(tree)
            gbm_dataframe = tree.df

    def make_tree(self, is_first, dataframe):
        """
        Create regression or classification tree.

        :param is_first: bool
            True if the first tree in the forest. The first tree uses the random seed, others use a
            randomly generated one
        :param dataframe: pandas DataFrame
            Dataframe that will initiate tree
        :return: DecisionTree
            DecisionTree that will be part of "forest"
        """
        if is_first:
            seed = self.random_seed
            dataframe = self.df
        else:
            seed = random.random()
            dataframe = dataframe

        if self.is_classification:  # placeholder, will contain true if/else once regression is implemented
            model = GBCTree(dataframe=dataframe,
                            y_col=self.y_col,
                            parent=None,
                            depth=0,
                            max_depth=self.max_depth,
                            min_sample_split=self.min_sample_split,
                            random_seed=seed,
                            gamma=self.gamma,
                            lambda_=self.lambda_,
                            previous_similarity=0)

        else:
            model = GBRTree(dataframe=dataframe,
                            y_col=self.y_col,
                            parent=None,
                            depth=0,
                            max_depth=self.max_depth,
                            min_sample_split=self.min_sample_split,
                            random_seed=seed,
                            gamma=self.gamma,
                            lambda_=self.lambda_,
                            previous_similarity=0)

        return model

    def predict(self, individual_value, cutoff = 0.5):
        if not self.tree_list:
            raise AttributeError("Model has not been fit, so predictions can be made.")

        prediction_li = []

        for tree in self.tree_list:
            if self.is_classification:
                prediction_li.append(tree.predict_proba(individual_value))
            else:
                prediction_li.append(tree.predict(individual_value))

        avg_prediction = np.mean(prediction_li)

        # For a classification problem, return 1 or 0 depending on value and cutoff
        if self.is_classification:
            if avg_prediction >= cutoff:
                avg_prediction = 1
            else:
                avg_prediction = 0

        return avg_prediction

    def predict_proba(self, individual_value):
        if not self.is_classification:
            raise AttributeError("predict_proba is not available for regression. Please use .predict instead")

        else:
            prediction_li = []

            for tree in self.tree_list:
                prediction_li.append(tree.predict_proba(individual_value))

        return np.mean(prediction_li)

if __name__ == '__main__':
    is_classification_0 = False
    print_trees = True

    df, individual_val, true_value = get_dataframe(is_classification_0)
    gbm = GBCModel(dataframe=df, y_col='y', max_depth=3, gamma = 2,
                   num_trees=3, random_seed=777, is_classification=is_classification_0)
    gbm.make_gbm()

    print('true value:', true_value, 'predicted value:', gbm.predict(individual_val))

    if is_classification_0:
        print('true value:', true_value, 'predicted value:', gbm.predict_proba(individual_val))

    if print_trees:
        for idx, gbm_tree in enumerate(gbm.tree_list):
            print('~~~TREE NUMBER {}~~~'.format(idx+1))
            print_breadth_first(gbm_tree)