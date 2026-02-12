from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from model.base import BaseModel
from model.model_evaluation import  evaluate


class XGBoostModel(BaseModel):
    @property
    def name(self) -> str:
        return "XGBoost"

    def build_model(self):
        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=7
    )


    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_prob = self.predict_proba(x_test)
        return evaluate(y_test, y_pred, y_prob)

