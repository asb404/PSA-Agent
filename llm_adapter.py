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
        # Very small deterministic 'generation' used for smoke tests.
        time.sleep(self._delay)
        return f"[DUMMY RESPONSE] prompt_len={len(prompt)} name={self._name}"

    def info(self) -> dict:
        return {"client": self._name, "type": "dummy", "delay": self._delay}

def try_local_llama(model_path: str = None) -> Tuple[LLMClient, str]:
    """Attempt to return a real llama-cpp-python client if available.
    Returns (client, message). If not available, returns (DummyLLMClient, reason).
    """
    try:
        # Try to import llama_cpp dynamically. If not installed, ImportError will be raised.
        from llama_cpp import Llama
        # Create a simple wrapper around Llama for our interface.
        class LocalLlamaClient(LLMClient):
            def __init__(self, model_path=None):
                self.model_path = model_path
                if model_path is None:
                    raise ValueError('model_path required for LocalLlamaClient')
                self._llm = Llama(model_path=model_path)

            def generate(self, prompt: str, **kwargs) -> str:
                # Map a few kwargs; keep it simple for the adapter.
                max_tokens = kwargs.get('max_tokens', 256)
                temperature = kwargs.get('temperature', 0.0)
                resp = self._llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                # llama-cpp-python returns a dict-like object with 'choices' key.
                text = resp.get('choices', [{}])[0].get('text', '')
                return text

            def info(self) -> dict:
                return {"client": "local_llama", "model_path": self.model_path}

        # If we got here, llama_cpp is available. But model_path might still be absent.
        return LocalLlamaClient(model_path=model_path), 'llama-cpp available'
    except Exception as e:
        # Fallback to Dummy client for environments without llama_cpp.
        return DummyLLMClient(), f'fallback to dummy client: {str(e)}'
