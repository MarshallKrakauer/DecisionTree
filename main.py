import pandas as pd
from math import sqrt, floor
import random
from sklearn.datasets import load_breast_cancer

data_bunch = load_breast_cancer()
cols = [c.replace(' ', '_') for c in data_bunch['feature_names']]
df = pd.DataFrame(data_bunch['data'], columns=cols)
df['y'] = data_bunch['target']

class DecisionNode:

    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0, max_depth=3):
        self.df = dataframe
        self.y_col = y_col
        self.depth = depth
        self.max_depth = max_depth
        self.parent = parent
        self.random_seed = random_seed
        self.left_child = None
        self.right_child = None
        self.gini = None
        self.best_column = None
        self.best_split = None

    def __str__(self):
        """
        Creates string value to represent node

        :return str: String with information about node
        """
        if self.gini is not None:
            gini_str = 'Gini: ' + str(round(self.gini,3))
            split_str = 'Split: ' + str(round(self.best_split, 3))
            size_str = 'Size: ' + str(len(self.df))
            depth_str = 'Depth: ' + str(self.depth)
            col_str = 'Feature: ' + self.best_column
            return col_str + ' ' + split_str +  ' ' + gini_str + ' ' + size_str + ' ' + depth_str
        else:
            return "terminal"

    @property
    def children(self):
        """List of node children"""
        return [self.left_child, self.right_child]

    def calculate_gini(self, column, threshold):
        """
        Calculate the gini impurity for a given split at a given column

        :param column: Column from which to make check split on
        :param threshold: number at which to split
        :return: gini score
        """
        gini = 0  # returning value

        # Split into two dataframes
        df_0 = self.df[self.df[column]  > threshold]
        df_1 = self.df[self.df[column] <= threshold]

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

    def get_split_list(self, column):
        """
        Get list of value from which to make split

        :param li column: column in data frame
        :return li: ordered list of unique values in dataframe
        """
        return list(set(self.df[column].values))

    def find_best_split(self):
        """
        For a given column, find the split with the lowest gini impurity

        :return tuple:
            best_column : column from which split is made
            best_split : value at which to split the best_column
            best_gini : gini value at given split
        """
        best_column = None  # column that provides best split
        best_split = None  # value which provides best split
        best_gini_value = float('inf')  # stores value for best split

        for col in self.select_columns():
            split_li = self.get_split_list(col)
            for split in split_li:
                current_gini_value = self.calculate_gini(col, split)
                if current_gini_value < best_gini_value:
                    best_gini_value = current_gini_value
                    best_column = col
                    best_split = split

        self.gini = best_gini_value
        self.best_column = best_column
        self.best_split = best_split
        return best_column, best_split, best_gini_value

    def select_columns(self):
        """Choose subset of columns of which to make splits

        :return list: list of columns which to check for splits
        """
        random.seed(self.random_seed)
        num_columns = floor(sqrt(len(self.df.columns)))
        features = [col for col in self.df.columns if col != self.y_col]

        col_list = random.sample(features,num_columns)
        return col_list

    def make_split(self, col, cutoff):
        """
        Make the two child nodes based on the given split

        :param col: feature which tree will be split on
        :param cutoff: value from which to make split in tree
        """
        self.left_child = DecisionNode(dataframe= self.df[self.df[col] > cutoff],
                              y_col=self.y_col,
                              parent=self,
                              depth=self.depth+1,
                              max_depth=self.max_depth,
                              random_seed=random.randint(1,1000000))

        self.right_child = DecisionNode(dataframe= self.df[self.df[col] <= cutoff],
                              y_col=self.y_col,
                              parent=self,
                              depth=self.depth+1,
                              max_depth=self.max_depth,
                              random_seed=random.randint(1,1000000))

    def create_child_nodes(self):
        """Splits data and creates two child nodes.

        May be unnecessary. Will consider refactor
        """
        best_column, best_split, _ = self.find_best_split()
        self.make_split(best_column, best_split)

    def create_tree(self):
        """Creates decision tree from a root node"""
        if self.depth <= self.max_depth:
            self.create_child_nodes()
            self.left_child.create_tree()
            self.right_child.create_tree()

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

dn = DecisionNode(df,'y')
dn.create_tree()
print_breadth_first(dn)
