"""In progress, tree will implement XGBoost Algorithm. File does not currently run"""

from math import sqrt, floor, log, exp
import random
from ClassificationTree import ClassificationTree, get_dataframe, print_breadth_first

class XGBCTree(ClassificationTree):

    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, min_impurity_decrease=float('-inf'),
                 gamma=1, lambda_ = 1, previous_similarity=0, eta = 0.3):

        if 'observation_probability__' not in dataframe.columns:
            dataframe['observation_probability__'] = 0.5

        super().__init__(dataframe, y_col, parent, depth, random_seed, max_depth,
                         min_sample_split, min_impurity_decrease, False)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.parent_similarity = previous_similarity
        self.eta = eta
        self.similarity = self.calculate_similarity(self.df)

    @property
    def prediction_log_odds(self):
        return [log_odds(p) for p in self.df['observation_probability__']]

    def calculate_similarity(self, dataframe):
        residual_sum = 0
        denominator = 0

        targets = dataframe[self.y_col].values
        current_probabilities = dataframe['observation_probability__'].values

        # Loop through dataframe to calculate similarity score
        for i in range(len(targets)):
            residual = targets[i] - current_probabilities[i]
            residual_sum += residual * residual
            denominator += current_probabilities[i] * (1-current_probabilities[i])

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
            score += self.calculate_similarity(split)

        # Make the score negative
        return self.similarity - score

    def make_split(self):
        """ Make the two child nodes based on the given split"""

        self.find_best_split()

        # If node is terminal, no split is made
        if self.is_terminal:
            return

        # Two smaller dataframes will become children of the trees
        # We will also create a subset of the row probabilities to send to the child trees
        left_child_df = self.df[self.df[self.best_column] > self.best_split]
        right_child_df = self.df[self.df[self.best_column] <= self.best_split]

        self.left_child = XGBCTree(dataframe=left_child_df,
                                   y_col=self.y_col,
                                   parent=self,
                                   depth=self.depth+1,
                                   max_depth=self.max_depth,
                                   min_sample_split=self.min_sample_split,
                                   min_impurity_decrease=self.min_impurity_decrease,
                                   random_seed=random.random(),
                                   gamma=self.gamma,
                                   lambda_ = self.lambda_,
                                   previous_similarity=self.parent_similarity)

        self.right_child = XGBCTree(dataframe=right_child_df,
                                    y_col=self.y_col,
                                    parent=self,
                                    depth=self.depth+1,
                                    max_depth=self.max_depth,
                                    min_sample_split=self.min_sample_split,
                                    min_impurity_decrease=self.min_impurity_decrease,
                                    random_seed=random.random(),
                                    gamma=self.gamma,
                                    lambda_ = self.lambda_,
                                    previous_similarity=self.parent_similarity)

    def select_columns(self):
        """Choose subset of columns of which to make splits

        :return list: list of columns which to check for splits
        """
        random.seed(self.random_seed)

        features = list(self.df.columns)
        features.remove(self.y_col)
        features.remove('observation_probability__')
        num_columns = floor(sqrt(len(features)))
        col_list = random.sample(features,num_columns)
        return col_list

    def update_probabilities(self):
        log_odds_li = self.prediction_log_odds
        tree_predictions = []

        for row_num in range(self.df.shape[0]):
            row_probability = self.predict_proba(self.df.iloc[row_num, :])
            row_log_odds = log_odds(row_probability)
            new_log_odds = log_odds_li[row_num]  + self.eta * row_log_odds
            new_prob = exp(new_log_odds) / (1 + exp(new_log_odds))
            tree_predictions.append(new_prob)

        self.df['observation_probability__'] = tree_predictions

def log_odds(probability):
    if probability == 1:
        return 100
    elif probability == 0:
        return -100
    else:
        return log(probability / (1-probability))


if __name__ == '__main__':
    # Testing right now. Code does not currently work
    df, individual_val, true_value = get_dataframe(True)
    dn = XGBCTree(df, 'y', random_seed=999, min_impurity_decrease=None, min_sample_split=-1)
    dn.create_tree()
    print_breadth_first(dn)
    # probability_0_ = dn.predict_proba(individual_val)
    # probability_1_ = dn.predict(individual_val)
    # print(probability_0_, probability_1_, true_value)
    dn.update_probabilities()