from collections import defaultdict
from ClassificationTree import ClassificationTree
from RegressionTree import RegressionTree
from MultiClassTree import MultiClassTree
from AbstractDecisionTree import print_breadth_first, get_dataframe, get_multi_class_dataframe
import random
import numpy as np

class RandomForest:

    def __init__(self, dataframe, y_col='target', target_type='binary', num_trees=3,
                 parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf')):
        if num_trees > 10:
            raise ValueError("Max of 10 trees")
        elif num_trees < 2:
            raise ValueError("At least 2 trees required")
        else:
            self.num_trees = int(num_trees)
        self.df = dataframe
        self.y_col = y_col
        self.depth = depth
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_impurity_decrease = min_impurity_decrease
        self.parent = parent
        self.random_seed = random_seed
        self.tree_list = []
        self.target_type = target_type

    def create_trees(self):
        """Initialize and fit all the trees in the random forest"""
        for i in range(self.num_trees):
            # For the first model, use the random seed. We don't want to use that seed for every model
            # Since it will produce identical copies of the first tree
            model = self.make_classifier(is_first=i==0)
            model.create_tree()
            self.tree_list.append(model)

    def make_classifier(self, is_first):
        """
        Create regression or classification tree.

        :param is_first: bool
            True if the first tree in the forest. The first tree uses the random seed, others use a
            randomly generated one
        :return: DecisionTree
            DecisionTree that will be part of "forest"
        """
        if is_first:
            seed = self.random_seed
        else:
            seed = random.random()

        if self.target_type == 'binary':
            model = ClassificationTree(self.df, self.y_col, None, 0, seed,
                                       self.max_depth, self.min_sample_split, self.min_impurity_decrease)
        elif self.target_type == 'multi_class':
            model = MultiClassTree(self.df, self.y_col, None, 0, seed,
                                       self.max_depth, self.min_sample_split, self.min_impurity_decrease)
        else:
            model = RegressionTree(self.df, self.y_col, None, 0, seed,
                                   self.max_depth, self.min_sample_split, self.min_impurity_decrease)

        return model

    def predict_proba(self, data_row):
        if self.target_type == 'continuous':
            raise AttributeError("predict_proba not available for regression model")

        if self.target_type == 'binary':
            prediction_list = []
            for decision_tree in self.tree_list:
                percentage = decision_tree.predict_proba(data_row)
                prediction_list.append(percentage)

            return np.mean(prediction_list)

        elif self.target_type == 'multi_class':
            def default_dict_zero():
                return 0
            predict_dict = defaultdict(default_dict_zero)
            for decision_tree in self.tree_list:
                output_dict = decision_tree.predict_proba(data_row)
                for key, percent in output_dict.items():
                    predict_dict[key] += percent / self.num_trees

            return dict(predict_dict)

    def predict(self, data_row, cutoff=0.5):
        if self.target_type == 'binary':
            if self.predict_proba(data_row) >= cutoff:
                return 1
            else:
                return 0

        elif self.target_type == 'continuous':
            prediction_list = []

            for decision_tree in self.tree_list:
                percentage = decision_tree.predict(data_row)
                prediction_list.append(percentage)

            return np.mean(prediction_list)

        else:
            prediction_dict = self.predict_proba(data_row)
            max_value = float('-inf')
            best_prediction = None
            for key, current_value in prediction_dict.items():
                if prediction_dict[key] > max_value:
                    max_value = current_value
                    best_prediction = key
            return best_prediction

if __name__ == '__main__':
    is_classification = False
    print_trees = False

    df, individual_val, true_value = get_multi_class_dataframe()  # get_dataframe(is_classification)
    rf = RandomForest(dataframe=df, y_col='y',target_type='multi_class',
                      max_depth=3, num_trees=3, random_seed=777, min_impurity_decrease=0.4)
    rf.create_trees()

    if print_trees:
        for idx, tree in enumerate(rf.tree_list):
            print('~~~TREE NUMBER {}~~~'.format(idx+1))
            print_breadth_first(tree)

    prob = rf.predict_proba(individual_val)
    value = rf.predict(individual_val)
    print(prob, value)
    # if is_classification:
    #     prob = rf.predict_proba(individual_val)
    #     class_ = rf.predict(individual_val)
    #     print('predicted:', np.round(prob, 3),',', class_, 'actual:', true_value)
    # else:
    #     value = rf.predict(individual_val)
    #     print('predicted:', np.round(value, 3), 'actual:', true_value)