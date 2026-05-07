from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def train(self, data):
        """Trains the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, steps):
        """Predicts the next 'steps' values."""
        pass

    @abstractmethod
    def evaluate(self, train_data, test_data):
        """Evaluates the model and returns metrics."""
        pass
