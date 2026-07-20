"""
llm_inference.py
================
Handles calling Qwen3 models via Ollama for classification.
"""

from __future__ import annotations
import requests


class LLMInference:
    """Wrapper for Qwen3 models via Ollama API."""

    def __init__(
        self,
        model_size: str,
        thinking_mode: bool = False,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_size = model_size
        self.model_name = f"qwen3:{model_size}"
        self.thinking_mode = thinking_mode
        self.base_url = base_url
        self.timeout = timeout
        self.last_thinking_trace: str | None = None
        self.last_response: str | None = None

    def load(self) -> None:
        """Check that the model is available in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            models = [m["name"] for m in response.json().get("models", [])]
            if self.model_name not in models:
                print(f"⚠️  Model '{self.model_name}' not found. Pulling...")
                requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name, "stream": False},
                    timeout=self.timeout,
                )
            print(f"Using Ollama model: {self.model_name} (thinking={self.thinking_mode})")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Cannot connect to Ollama. Run: ollama serve")

    def classify(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Run inference using Ollama generate API.
        `think` explicitly toggles Qwen3's reasoning mode.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": "Output ONLY the classification label. No explanation.",
            "stream": False,
            "think": self.thinking_mode,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.0,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s "
                f"(model={self.model_name}, thinking={self.thinking_mode})."
            )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.text}")

        result = response.json()
        
        # Store both response and thinking trace
        self.last_response = result.get("response", "").strip()
        self.last_thinking_trace = result.get("thinking")
        
        # Print thinking trace if available
        if self.thinking_mode and self.last_thinking_trace:
            print(f"\n🧠 Thinking Trace:\n{self.last_thinking_trace}\n")
        
        # Print token usage
        print(f"\n🔢 Tokens: prompt_eval={result.get('prompt_eval_count')}, "
              f"eval={result.get('eval_count')}, "
              f"done_reason={result.get('done_reason')}")

        content = result.get("response", "").strip()
        content = content.strip('"\'/ \n')

        return content

    def unload(self) -> None:
        """Nothing to unload — Ollama manages models server-side."""
        pass