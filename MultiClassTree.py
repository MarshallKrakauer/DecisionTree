"""Working file for multi-class classification.

At some point, I may integrate this functionality into the ClassificationTree"""

from sklearn.datasets import load_wine
import pandas as pd
import random
from ClassificationTree import ClassificationTree, print_breadth_first


class MultiClassTree(ClassificationTree):

    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf'), bootstrap=True, gamma=None):

        super().__init__(dataframe, y_col, parent, depth, random_seed, max_depth,
                     min_sample_split, min_impurity_decrease,bootstrap,gamma)

    @property
    def probability(self):
        """
        Produce dictionary of potential probabilities for each class

        :return: dict
            Dictionary of probabilities for each class
        """
        probability_dict = {}
        temp_df = self.df.sort_values(self.y_col, ascending=True)

        for class_ in temp_df[self.y_col].unique():
            probability_dict[class_] = len(temp_df[temp_df[self.y_col] == class_]) / len(temp_df)

        return probability_dict

    def calculate_split_criterion(self, column, threshold):
        """
        Calculate the gini impurity for a given split at a given column

        :param column: Column from which to make check split on
        :param threshold: number at which to split
        :return: gini score
        """
        gini = 0  # returning value
        class_list = self.df[self.y_col].unique()

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
                for class_ in class_list:
                    probability_score = len(split[split[self.y_col] == class_]) / len(split)
                    temp_score += probability_score * probability_score

            gini += (1 - temp_score) * (len(split) / len(self.df))

        return gini

    def make_split(self):
        """
        Make the two child nodes based on the given split
        """

        self.find_best_split()

        # If node is terminal, no split is made
        if self.is_terminal:
            return

        self.left_child = MultiClassTree(dataframe=self.df[self.df[self.best_column] > self.best_split],
                                             y_col=self.y_col,
                                             parent=self,
                                             depth=self.depth+1,
                                             max_depth=self.max_depth,
                                             min_sample_split=self.min_sample_split,
                                             min_impurity_decrease=self.min_impurity_decrease,
                                             random_seed=random.random())

        self.right_child = MultiClassTree(dataframe= self.df[self.df[self.best_column] <= self.best_split],
                                              y_col=self.y_col,
                                              parent=self,
                                              depth=self.depth+1,
                                              max_depth=self.max_depth,
                                              min_sample_split=self.min_sample_split,
                                              min_impurity_decrease=self.min_impurity_decrease,
                                              random_seed=random.random())
    def predict_proba(self, data_row):
        if self.is_terminal:
            return self.probability
        else:
            value = data_row[self.best_column]
            if value > self.best_split:
                return self.left_child.predict_proba(data_row)
            else:
                return self.right_child.predict_proba(data_row)


    def predict(self, data_row, cutoff = None):
        outcome_dict = self.predict_proba(data_row)
        max_value = float('-inf')
        best_prediction = None

        for key, value in outcome_dict.items():
            if value > max_value:
                max_value = value
                best_prediction = key

        return best_prediction

if __name__ == '__main__':
    data_bunch = load_wine()
    df = pd.DataFrame(data_bunch['data'], columns=data_bunch['feature_names'])
    df.rename(columns={'od280/od315_of_diluted_wines': 'diluted'}, inplace=True)
    individual_value = df.iloc[10, :]
    df['y'] = data_bunch['target']
    mct = MultiClassTree(df, 'y', max_depth=3, random_seed=777, min_sample_split=10)
    mct.create_tree()
    print_breadth_first(mct)
    print('probability_dict:', mct.predict_proba(individual_value), 'prediction:', mct.predict(individual_value))

