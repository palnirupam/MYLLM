from abc import ABC, abstractmethod
from typing import Generator

class InferenceRuntime(ABC):
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        ...

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        ...

    @abstractmethod
    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        """Generator yielding tokens one at a time."""
        ...
