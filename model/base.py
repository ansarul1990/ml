from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseModel(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def build_model(self):
        ...

    def __init__(self):
        self.model = self.build_model()

    def fit(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        raise NotImplementedError(f"{self.name} does not support predict_proba()")

    @abstractmethod
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        ...
