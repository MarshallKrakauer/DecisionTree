import pandas as pd
from math import sqrt, floor
import random
from sklearn.datasets import load_breast_cancer

data_bunch = load_breast_cancer()
cols = [c.replace(' ', '_') for c in data_bunch['feature_names']]
df = pd.DataFrame(data_bunch['data'], columns=cols)
df['y'] = data_bunch['target']

class DecisionNode:

    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0):
        self.df = dataframe
        self.y_col = y_col
        self.depth = depth
        self.parent = parent
        self.children = None
        self.random_seed = random_seed

    def calculate_gini(self, column, threshold):
        gini = 0  # returning value

        # Split into two dataframes
        df_0 = self.df[self.df[column]  > threshold]
        df_1 = self.df[self.df[column] <= threshold]

        for split in [df_0, df_1]:
            temp_score = 0
            if len(split) > 0:
                probability_0 = len(split[split[self.y_col] == 0]) / len(split)
                probability_1 = len(split[split[self.y_col] == 1]) / len(split)
                temp_score += (probability_0 * probability_0 + probability_1 * probability_1)
            gini += (1 - temp_score) * (len(split) / len(self.df))

        return gini

    def get_split_list(self, column):
        value_li = self.df[column].values
        value_li.sort()
        return value_li

    def find_best_split(self):
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

        return best_column, best_split, best_gini_value

    def select_columns(self):
        random.seed(self.random_seed)
        num_columns = floor(sqrt(len(self.df.columns)))
        features = [col for col in self.df.columns if col != self.y_col]

        col_list = random.sample(features,num_columns)
        return col_list

    def make_split(self, col, cutoff):
        node_0 = DecisionNode(dataframe= self.df[self.df[col] > cutoff],
                              y_col=self.y_col,
                              parent=self,
                              depth=self.depth+1,
                              random_seed=self.random_seed)

        node_1 = DecisionNode(dataframe= self.df[self.df[col] <= cutoff],
                              y_col=self.y_col,
                              parent=self,
                              depth=self.depth+1,
                              random_seed=self.random_seed)

        self.children = [node_0, node_1]

    def create_child_nodes(self):
        best_column, best_split, _ = self.find_best_split()
        self.make_split(best_column, best_split)

class DecisionTree:

    def __init__(self,root_node,max_depth=3):
        self.root_node = root_node
        self.max_depth = max_depth

dn = DecisionNode(df, 'y')
print(dn.find_best_split())