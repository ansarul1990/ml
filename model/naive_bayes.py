from sklearn.naive_bayes import GaussianNB

from model.base import BaseModel
from model.metrices import  compute_metrics


class NaiveBayesModel(BaseModel):
    @property
    def name(self) -> str:
        return "Naive Bayes"

    def build_model(self):
        return GaussianNB()


    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_prob = self.predict(x_test)
        return compute_metrics(y_test, y_pred, y_prob)

