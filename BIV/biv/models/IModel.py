from abc import ABC, abstractmethod

class IModel(ABC):
    @abstractmethod
    def train(self, DTR, LTR, save_file_name = None):
        pass

    @abstractmethod
    def test(self, DTE):
        pass

    @staticmethod
    @abstractmethod
    def load(file_name):
        pass