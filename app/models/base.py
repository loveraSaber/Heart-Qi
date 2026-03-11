from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
