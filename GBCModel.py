"""Gradient Boosting Classier Model. It currently runs, but the trees don't work properly"""

import random
from GBCTree import GBCTree
from AbstractDecisionTree import print_breadth_first, get_dataframe


class GBCModel:

    def __init__(self, dataframe, y_col='target', random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf'),
                 gamma=1, lambda_=1, previous_similarity=0, eta=0.3, num_trees=3):
        self.df = dataframe
        self.y_col = y_col
        self.random_seed = random_seed
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_impurity_decrease = min_impurity_decrease
        self.gamma = gamma
        self.lambda_ = lambda_
        self.parent_similarity = previous_similarity
        self.eta = eta
        self.num_trees = num_trees
        self.tree_list = []

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

        if True:  # placeholder, will contain true if/else once regression is implemented
            model = GBCTree(dataframe=dataframe,
                            y_col=self.y_col,
                            parent=None,
                            depth=0,
                            max_depth=self.max_depth,
                            min_sample_split=self.min_sample_split,
                            min_impurity_decrease=self.min_impurity_decrease,
                            random_seed=seed,
                            gamma=self.gamma,
                            lambda_=self.lambda_,
                            previous_similarity=0)

        return model

if __name__ == '__main__':
    is_classification = True
    print_trees = True

    df, individual_val, true_value = get_dataframe(is_classification)
    gbm = GBCModel(dataframe=df, y_col='y', max_depth=3, min_impurity_decrease=None, num_trees=3, random_seed=777)
    gbm.make_gbm()

    if print_trees:
        for idx, gbm_tree in enumerate(gbm.tree_list):
            print('~~~TREE NUMBER {}~~~'.format(idx+1))
            print_breadth_first(gbm_tree)