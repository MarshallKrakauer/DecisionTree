# DecisionTree
Custom implementation of a tree-bassed machine learning algorithms. 

CART (classificaiton and regression tree) models form two of the most popular machine learning algorithms: random forests and gradiet boosting. Thus, I want to gain a deeper understanding of how these models work. To do so, I've decided to implement the models from scratch.

This repository has a lot of files. I will explain them here:

1. AbstractDecisionTree.py - File is not run directly. Creates the base functions that are inherited by the ClassificationTree and RegressionTree
2. ClassificationTree.py - Decision tree for data with 0 or 1 target variable 
3. MultiClassTree.py - Classification Tree for data with more than 2 classes. Currently not supported by Random Forest or Gradient Boosting
4. RegresionTree.py - Decision tree for data with continuous target variable
5. RandomForest.py - Creates random set of either Classification or Regression Tree models.
6. GBCTree.py - Inherits ClassificationTree.py. Creates custom classification tree for a gradient boosting tree algorithm
7. GBRTree.py - Inherits RegressionTree.py. Creates custom regression tree for a gradient boosting tree algorithm
8. GradientBoostingModel.py - Creates boosting model using either GBCTree or GBRTree.

Note: The model encodes categorical variables. This might not be appropriate for some uses cases, but it matches the functionality of sklearn. 
