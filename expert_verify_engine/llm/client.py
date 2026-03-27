import time
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from expert_verify_engine.app.config import get_config
from expert_verify_engine.utils.retry import default_retry


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(
        self,
        logger: Callable[[str, str, str, dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.api_key = get_config("api_key")
        self.model = get_config("model")
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY not set")

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.logger = logger

    @default_retry
    def chat(
        self,
        prompt: str,
        temperature: float | None = None,
        prompt_type: str | None = None,
    ) -> str:
        if temperature is None:
            temperature = get_config("temperature")

        start_time = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            result = response.choices[0].message.content

            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = (
                response.usage.completion_tokens if response.usage else 0
            )
            total_tokens = response.usage.total_tokens if response.usage else 0
            tokens_per_second = (
                completion_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
            )

            if self.logger:
                metadata = {
                    "temperature": temperature,
                    "prompt_type": prompt_type,
                    "timing": {
                        "wait_time_ms": elapsed_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "tokens_per_second": tokens_per_second,
                    },
                }
                self.logger(self.model, prompt, result, metadata)

            return result
        except Exception as e:
            raise LLMError(f"OpenAI API error: {e}") from e
