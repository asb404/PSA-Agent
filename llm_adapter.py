import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class LLMClient(ABC):
    '''
    Abstract LLM client interface.
    Implementations must provide generate(prompt, **kwargs) -> str
    and optionally info() -> dict for diagnostics.
    '''
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def info(self) -> dict:
        pass

class DummyLLMClient(LLMClient):
    """A deterministic fallback client used when real bindings are not available."""
    def __init__(self, name='dummy', delay=0.05):
        self._name = name
        self._delay = delay

    def generate(self, prompt: str, **kwargs) -> str:
        time.sleep(self._delay)
        return f"[DUMMY RESPONSE] prompt_len={len(prompt)} name={self._name}"

    def info(self) -> dict:
        return {"client": self._name, "type": "dummy", "delay": self._delay}

def try_local_llama(model_path: str = None) -> Tuple[LLMClient, str]:
    """Attempt to return a real llama-cpp-python client if available.
    Returns (client, message). If not available, returns (DummyLLMClient, reason).
    """
    try:
        from llama_cpp import Llama
        class LocalLlamaClient(LLMClient):
            def __init__(self, model_path=None):
                self.model_path = model_path
                if model_path is None:
                    raise ValueError('model_path required for LocalLlamaClient')
                self._llm = Llama(model_path=model_path)

            def generate(self, prompt: str, **kwargs) -> str:
                max_tokens = kwargs.get('max_tokens', 256)
                temperature = kwargs.get('temperature', 0.0)
                resp = self._llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                text = resp.get('choices', [{}])[0].get('text', '')
                return text

            def info(self) -> dict:
                return {"client": "local_llama", "model_path": self.model_path}

        return LocalLlamaClient(model_path=model_path), 'llama-cpp available'
    except Exception as e:
        return DummyLLMClient(), f'fallback to dummy client: {str(e)}'
