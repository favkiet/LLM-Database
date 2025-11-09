import os
import requests
from openai import OpenAI


class BaseLLM:
    """Base LLM interface."""

    def invoke(self, messages):
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    """OpenAI API wrapper (Pythonic, not using LangChain)."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key or self.api_key.startswith("${"):
            raise ValueError("Missing OPENAI_API_KEY. Please set it in environment.")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages):
        """messages: list of dict(role, content)"""
        formatted_msgs = [
            {"role": m.get("role"), "content": m.get("content")} for m in messages
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return {
            "role": "assistant",
            "content": response.choices[0].message.content.strip(),
        }


class OllamaLLM(BaseLLM):
    """Ollama HTTP API wrapper."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def invoke(self, messages):
        """messages: list of dict(role, content)"""
        formatted_msgs = [
            {"role": m.get("role"), "content": m.get("content")} for m in messages
        ]
        payload = {
            "model": self.model_name,
            "messages": formatted_msgs,
            "temperature": self.temperature,
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "role": "assistant",
            "content": data["message"]["content"].strip(),
        }

