from abc import ABC, abstractmethod


class VLMBackend(ABC):

    @abstractmethod
    def describe(self, image_path):
        pass
