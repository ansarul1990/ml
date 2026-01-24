
from sklearn.tree import DecisionTreeClassifier

from model.base import BaseModel
from model.metrices import  compute_metrics


class DecisionTreeModel(BaseModel):
    @property
    def name(self) -> str:
        return "Decision Tree"

    def build_model(self):
        return DecisionTreeClassifier(random_state=42)


    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_prob = self.predict_proba(x_test)
        return compute_metrics(y_test, y_pred, y_prob)

