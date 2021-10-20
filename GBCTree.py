"""Tree for Gradient Boosting Model. Currently missing lambda implementation, but the rest functions."""

from math import sqrt, floor, log, exp
import random
from ClassificationTree import ClassificationTree, get_dataframe, print_breadth_first


class GBCTree(ClassificationTree):

    def __init__(self, dataframe, y_col='target', parent=None, depth=0, random_seed=0.0, max_depth=3,
                 min_sample_split=0, gamma=1, lambda_=1, previous_similarity=0, eta=0.3):

        if 'observation_probability__' not in dataframe.columns:
            dataframe['observation_probability__'] = 0.5

        super().__init__(dataframe, y_col, parent, depth, random_seed, max_depth,
                         min_sample_split, None, False, gamma)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.parent_similarity = previous_similarity
        self.eta = eta
        self.similarity = self.calculate_similarity(self.df)

    def __str__(self):
        """
        Creates string value to represent node

        :return str: String with information about node
        """
        depth_str = 'Depth: ' + str(self.depth)

        # Internal or leaf node
        if self.is_terminal:
            terminal_str = 'Leaf, '
        else:
            terminal_str = 'Int, '

        # Different outputs for internal or leaf node
        if self.split_criterion is not None:
            gini_str = 'Gain: ' + str(abs(round(self.split_criterion, 3)))
            split_str = ' at ' + str(round(self.best_split, 3))
            size_str = 'Size: ' + str(len(self.df))
            col_str = 'Feature: ' + self.best_column
            return terminal_str + size_str + ' ' + depth_str + ' ' +  gini_str + ' ' + col_str +  split_str
        else:
            size_str = 'Size: ' + str(len(self.df))
            prob_str = 'Prob: ' + str(round(self.probability,3))
            return terminal_str + size_str + ' ' + depth_str + ' ' + prob_str

    @property
    def prediction_log_odds(self):
        """Calculate log odds for each current probability prediction."""
        return [log_odds(p) for p in self.df['observation_probability__']]

    @property
    def output_value(self):
        """Output value for each node in the XGBTree

        :return: float: output value used for boosting value
                like the similarity score, but without the numerator being squared
        """
        residual_sum = 0
        denominator = 0

        targets = self.df[self.y_col].values
        current_probabilities = self.df['observation_probability__'].values

        # Loop through dataframe to calculate similarity score
        for i in range(len(targets)):
            residual_sum += targets[i] - current_probabilities[i]
            denominator += current_probabilities[i] * (1 - current_probabilities[i])

        denominator += self.lambda_

        return residual_sum / denominator

    def calculate_similarity(self, dataframe):
        """Similarity score for each node and potential node

        :param dataframe: potential dataframe to check for similarity score
        :return: float: similarity score
        """
        residual_sum = 0
        denominator = 0

        targets = dataframe[self.y_col].values
        current_probabilities = dataframe['observation_probability__'].values

        # Loop through dataframe to calculate similarity score
        for i in range(len(targets)):
            residual_sum += targets[i] - current_probabilities[i]
            denominator += current_probabilities[i] * (1 - current_probabilities[i])

        denominator += self.lambda_

        return (residual_sum * residual_sum) / denominator

    def calculate_split_criterion(self, column, threshold):
        """Calculates gain, which is criteria used to make best split

        :param column: potential dataframe to check for similarity score
        :param threshold: potential dataframe to check for similarity score
        :return: float: gain in similarity from the parent node
        """
        score = 0  # returning value

        # Split into two dataframes
        df_0 = self.df[self.df[column] > threshold]
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

        self.left_child = GBCTree(dataframe=left_child_df,
                                  y_col=self.y_col,
                                  parent=self,
                                  depth=self.depth + 1,
                                  max_depth=self.max_depth,
                                  min_sample_split=self.min_sample_split,
                                  random_seed=random.random(),
                                  gamma=self.gamma,
                                  lambda_=self.lambda_,
                                  previous_similarity=self.similarity,
                                  eta=self.eta)

        self.right_child = GBCTree(dataframe=right_child_df,
                                   y_col=self.y_col,
                                   parent=self,
                                   depth=self.depth + 1,
                                   max_depth=self.max_depth,
                                   min_sample_split=self.min_sample_split,
                                   random_seed=random.random(),
                                   gamma=self.gamma,
                                   lambda_=self.lambda_,
                                   previous_similarity=self.similarity,
                                   eta=self.eta)

    def select_columns(self):
        """Choose subset of columns of which to make splits

        :return list: list of columns which to check for splits
        """
        random.seed(self.random_seed)

        features = list(self.df.columns)
        features.remove(self.y_col)
        features.remove('observation_probability__')
        num_columns = floor(sqrt(len(features)))
        col_list = random.sample(features, num_columns)
        return col_list

    def update_probabilities(self):
        """Update the dataframe probabilities using the gradient boosting algorithms."""
        log_odds_li = self.prediction_log_odds
        tree_predictions = []

        for row_num in range(self.df.shape[0]):
            row_output_value = self.get_output_value(self.df.iloc[row_num, :])
            new_log_odds = log_odds_li[row_num] + (self.eta * row_output_value)
            new_prob = exp(new_log_odds) / (1 + exp(new_log_odds))
            tree_predictions.append(new_prob)

        self.df['observation_probability__'] = tree_predictions

    def get_output_value(self, data_row):
        """Obtain output value (one used by gradient boosting algorithm) for a given row of data

        Recursive function. Searches for a terminal node.

        :param data_row: series from which to produce the output value.
        """
        if self.is_terminal:
            return self.output_value
        else:
            value = data_row[self.best_column]
            if value > self.best_split:
                return self.left_child.get_output_value(data_row)
            else:
                return self.right_child.get_output_value(data_row)

    def prune_tree(self):
        if self.max_depth <= 1:
            self.left_child.prune_tree()
            self.right_child.prune_tree()
        else:
            if abs(self.split_criterion) < self.gamma:
                self.is_terminal = True
                self.left_child = None
                self.right_child = None


def log_odds(probability):
    """Calculate the log odds for a given number.
    To prevent an undefined value (div by 0 or log(0)), I impute 100 and -100 for undefined values.
    """
    if probability == 1:
        return 100
    elif probability == 0:
        return -100
    else:
        return log(probability / (1 - probability))


if __name__ == '__main__':
    # Testing right now. Code does not currently work
    df, individual_val, true_value = get_dataframe(True)
    dn = GBCTree(df, 'y', random_seed=999, min_sample_split=-1, gamma=-999)
    dn.create_tree()
    print_breadth_first(dn)
    probability_0_ = dn.predict_proba(individual_val)
    probability_1_ = dn.predict(individual_val)
    #print(probability_0_, probability_1_, true_value)
    dn.update_probabilities()
