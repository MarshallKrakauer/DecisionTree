"""In progress, tree will implement XGBoost Algorithm"""

from math import sqrt, floor
import random
from ClassificationTree import ClassificationTree, get_dataframe, print_breadth_first

class XGBCTree(ClassificationTree):

    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf'),
                 gamma=1, lambda_ = 1, previous_prob=0.5, previous_similarity=0.5):
        super().__init__(dataframe, y_col, parent, depth, random_seed, max_depth,
                         min_sample_split, min_impurity_decrease)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.previous_prob = previous_prob
        self.parent_similarity = previous_similarity
        self.similarity = self.calculate_similarity(self.df, self.previous_prob)

    def calculate_similarity(self, dataframe, previous_prob):
        residual_sum = 0
        denominator = 0

        for i in range(len(dataframe)):
            residual = dataframe.loc[i, self.y_col] - previous_prob
            residual_sum += residual * residual
            denominator += previous_prob * (1-previous_prob)

        denominator += self.lambda_

        return residual_sum / denominator

    def calculate_split_criterion(self, column, threshold):
        score = 0  # returning value

        # Split into two dataframes
        df_0 = self.df[self.df[column]  > threshold]
        len_0 = len(df_0)
        df_1 = self.df[self.df[column] <= threshold]
        len_1 = len(df_1)

        if len_0 < self.min_sample_split or len_1 < self.min_sample_split:
            return float('-inf')

        # Calculate gain score for each dataframe
        for split in [df_0, df_1]:
            score += self.calculate_similarity(split, self.previous_prob)

        return score - self.similarity

    def make_split(self):
        """ Make the two child nodes based on the given split"""

        self.find_best_split()

        # If node is terminal, no split is made
        if self.is_terminal:
            return

        self.left_child = XGBCTree(dataframe=self.df[self.df[self.best_column] > self.best_split],
                                             y_col=self.y_col,
                                             parent=self,
                                             depth=self.depth+1,
                                             max_depth=self.max_depth,
                                             min_sample_split=self.min_sample_split,
                                             min_impurity_decrease=self.min_impurity_decrease,
                                             random_seed=random.random(),
                                             gamma=self.gamma,
                                             lambda_ = self.lambda_,
                                             previous_prob=self.previous_prob,
                                             previous_similarity=self.previous_similarity)

        self.right_child = XGBCTree(dataframe= self.df[self.df[self.best_column] <= self.best_split],
                                              y_col=self.y_col,
                                              parent=self,
                                              depth=self.depth+1,
                                              max_depth=self.max_depth,
                                              min_sample_split=self.min_sample_split,
                                              min_impurity_decrease=self.min_impurity_decrease,
                                              random_seed=random.random(),
                                              gamma=self.gamma,
                                              lambda_ = self.lambda_,
                                              previous_prob=self.previous_prob,
                                              previous_similarity=self.previous_similarity)

if __name__ == '__main__':
    # Testing right now. Code does not currently work
    df, individual_val, true_value = get_dataframe(True)
    dn = XGBCTree(df, 'y', random_seed=777)
    dn.create_tree()
    print_breadth_first(dn)
    probability_0_ = dn.predict_proba(individual_val)
    probability_1_ = dn.predict(individual_val)
    print(probability_0_, probability_1_, true_value)