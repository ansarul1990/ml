from model.decision_tree import DecisionTreeModel
from model.knn import KNNModel
from model.logistic import LogisticModel
from model.naive_bayes import NaiveBayesModel
from model.random_forest import RandomForestModel
from model.xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "Logistic Regression": LogisticModel,
    "Decision Tree": DecisionTreeModel,
    "KNN": KNNModel,
    "Naive Bayes": NaiveBayesModel,
    "Random Forest": RandomForestModel,
    "XGBOOST" : XGBoostModel,
}