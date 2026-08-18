from .config import ModelConfig

__all__ = ["ModelConfig", "MyLLMModel"]


def __getattr__(name):
    if name == "MyLLMModel":
        # Keep static config/tokenizer tooling usable without importing torch.
        from .transformer import MyLLMModel
        return MyLLMModel
    raise AttributeError(name)
