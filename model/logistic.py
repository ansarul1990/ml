
from sklearn.linear_model import LogisticRegression
from model.base import BaseModel
from model.model_evaluation import  evaluate


class LogisticModel(BaseModel):
    @property
    def name(self) -> str:
        return "Logistic Regression"

    def build_model(self):
        return LogisticRegression(max_iter=1000)


    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_prob = self.predict_proba(x_test)
        return evaluate(y_test, y_pred, y_prob)

