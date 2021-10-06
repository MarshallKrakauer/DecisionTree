"""In progress, tree will implement XGBoost Algorithm"""

from math import sqrt, floor
import random
from ClassificationTree import ClassificationTree

class XGBCTree(ClassificationTree):
    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf'),
                 gamma=1, lambda_ = 1, previous_prob=0.5):
        super().__init__(dataframe, y_col, parent, depth, random_seed, max_depth,
                         min_sample_split, min_impurity_decrease)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.previous_prob = previous_prob

    @property
    def similarity_score(self):
        residual_sum = 0
        denominator = 0

        for i in range(len(self.dataframe)):
            residual = self.dataframe.loc[i, self.y_col] - self.previous_prob
            residual_sum += residual * residual
            denominator += self.previous_prob * (1-self.previous_prob)

        denominator += self.lambda_

        return residual_sum / denominator