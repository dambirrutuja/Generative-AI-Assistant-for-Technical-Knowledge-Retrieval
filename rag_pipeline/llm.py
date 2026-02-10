from __future__ import annotations
import os
from typing import Optional, Dict, Any

class BaseLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

class LocalHFLLM(BaseLLM):
    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.2):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt: str, **kwargs) -> str:
        import torch
        max_new_tokens = int(kwargs.get("max_new_tokens", self.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(0.01, temperature),
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        # Uses the new OpenAI python library if installed; otherwise raises.
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

    def generate(self, prompt: str, **kwargs) -> str:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":prompt},
            ],
            temperature=float(kwargs.get("temperature", 0.2)),
        )
        return resp.choices[0].message.content.strip()

def build_llm(cfg: Dict[str, Any]) -> BaseLLM:
    provider = cfg["llm"]["provider"].lower()
    if provider == "openai":
        return OpenAILLM()
    # default local
    return LocalHFLLM(
        model_name=cfg["llm"]["local_model_name"],
        max_new_tokens=int(cfg["llm"]["max_new_tokens"]),
        temperature=float(cfg["llm"]["temperature"]),
    )
