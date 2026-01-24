from sklearn.ensemble import RandomForestClassifier

from model.base import BaseModel
from model.metrices import  compute_metrics


class RandomForestModel(BaseModel):
    @property
    def name(self) -> str:
        return "Random Forest"

    def build_model(self):
        return RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1
        )


    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_prob = self.predict_proba(x_test)
        return compute_metrics(y_test, y_pred, y_prob)

