"""Functions that can be used to view the decision tree splits"""
from abc import abstractmethod
from sklearn.datasets import load_breast_cancer, load_diabetes
from math import sqrt, floor
import random
import pandas as pd

class DecisionTree:

    """
    Single node of decision tree that finds best split for a given dataframe.

    Attributes
    ----------
    df : pandas DatFrame
        Dataframe to split based on gini impurity
    y_col : str
        Name of column that contains 1/0 target variable
    depth : int
        Number of parents of current node
    max_depth : int
        Maximum layers of descendants for tree's root node
    parent : ClassificationTree
        Parent of current node, split that lead to this node
    random_seed : int
        Random value used for tree
    left_child : ClassificationTree
        Node based on best split
    right_child : ClassificationTree
        Other node based on best split
    split_criterion : float
        Gini value of the best possible split
    best_column : string
        Name of column from which data is split
    best_split : float
        Value from which the best column is split on
    """
    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf'), bootstrap=True, gamma=None):
        self.df = dataframe
        self.y_col = y_col
        self.depth = depth
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_impurity_decrease = min_impurity_decrease
        self.parent = parent
        self.random_seed = random_seed
        self.bootstrap = bootstrap
        self.left_child = None
        self.right_child = None
        self.split_criterion = None
        self.best_column = None
        self.best_split = None
        self.is_terminal = False
        self.gamma = gamma

        if self.bootstrap:
            self.df = self.df.sample(frac=1, replace=True,random_state=int(self.random_seed))

        # Check to make sure dataframe has valid types
        self.check_dataframe_dtypes()

        # Check to see if node is terminal
        if self.depth == self.max_depth:
            self.is_terminal = True

        if len(self.df) < self.min_sample_split:
            self.is_terminal = True

    def check_dataframe_dtypes(self):
        """Return error if dataframes aren't object or numeric."""
        for col in self.df.columns:
            data_type = self.df[col].dtype
            if data_type == 'object':
                self.encode_column(col)
            if data_type not in ['int64', 'float64', 'int32', 'float32', 'object']:
                raise ValueError('Columns must be integer, float, or object')

    def encode_column(self, categorical_column):
        """
        Converts categorical column to an integer column.

        :param categorical_column: str
            Name of object column to be encoded
        """
        value_dict = {}
        unique_values = self.df[categorical_column].unique()
        for idx, category in enumerate(unique_values):
            value_dict[category] = idx

        self.df[categorical_column] = self.df[categorical_column].apply(lambda x: value_dict[x])

    @abstractmethod
    def __str__(self):
        pass

    @property
    def children(self):
        """List of node children"""
        return [self.left_child, self.right_child]

    @abstractmethod
    def calculate_split_criterion(self, column, threshold):
        pass

    def get_split_list(self, column):
        """
        Get list of value from which to make split

        :param li column: column in data frame
        :return li: ordered list of unique values in dataframe
        """
        return list(set(self.df[column].values))

    def select_columns(self):
        """Choose subset of columns of which to make splits

        :return list: list of columns which to check for splits
        """
        random.seed(self.random_seed)

        features = [col for col in self.df.columns if col != self.y_col]
        features.remove(self.y_col)
        num_columns = floor(sqrt(len(features)))
        col_list = random.sample(features,num_columns)
        return col_list

    def find_best_split(self):
        """
        For a given column, find the split with the lowest gini impurity
        """
        best_column = None  # column that provides best split
        best_split = None  # value which provides best split
        best_split_value = float('inf') # stores value for best split

        for col in self.select_columns():
            # Get the list of splits for each column and find the best gini value
            split_li = self.get_split_list(col)
            for split in split_li:
                current_split_value = self.calculate_split_criterion(col, split)
                if current_split_value <= best_split_value:
                    best_split_value = current_split_value
                    best_column = col
                    best_split = split

        # todo Add a min impurity gain
        if self.parent is not None:
            parent_split_val = self.parent.split_criterion
        else:
            parent_split_val = float('inf')

        impurity_decrease = parent_split_val - best_split_value
        if ((self.min_impurity_decrease is None and self.gamma is None)
                or (self.gamma is None and impurity_decrease > self.min_impurity_decrease)
                or (self.min_impurity_decrease is None and abs(best_split_value) > self.gamma)):
            self.split_criterion = best_split_value
            self.best_column = best_column
            self.best_split = best_split
        else:
            self.is_terminal = True

    @abstractmethod
    def make_split(self):
        pass

    def create_tree(self):
        """Creates decision tree from a root node"""
        self.make_split()
        if not self.is_terminal:
            self.left_child.create_tree()
            self.right_child.create_tree()

    @abstractmethod
    def predict(self, data_row):
        pass


def print_current_level(node, level):
    """
    Prints all nodes at the given level of the tree

    :param node: root node from which to check for values
    :param level: depth of tree (starting at 0) from which to print
    """
    if level < 0:
        raise ValueError("minimum depth is 0")

    if node is None:
        return
    if level == 0:
        print(node)
    elif level > 0:
        print_current_level(node.left_child, level-1)
        print_current_level(node.right_child, level-1)

def print_breadth_first(node):
    """
    Print the nodes depth first (starting from 0 and going to the bottom).

    :param node: root_node of decision tree
    """
    for i in range(0, node.max_depth+1):
        print_current_level(node,i)

def get_dataframe(classification=True):
    if classification:
        data_bunch = load_breast_cancer()
    else:
        data_bunch = load_diabetes()

    cols = [c.replace(' ', '_') for c in data_bunch['feature_names']]
    df = pd.DataFrame(data_bunch['data'], columns=cols)
    df['y'] = data_bunch['target']
    individual_val = df.loc[0, df.columns != 'y']
    true_value = df.loc[0, 'y']

    return df, individual_val, true_value

