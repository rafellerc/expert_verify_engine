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

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            result = response.choices[0].message.content

            wait_time_ms = 0
            tokens = 0
            tokens_per_second = 0.0

            if response.model_extra:
                extra = response.model_extra
                if "latency" in extra:
                    wait_time_ms = int(extra["latency"] * 1000)
                if "usage" in extra:
                    usage = extra["usage"]
                    if "completion_tokens" in usage:
                        tokens = usage["completion_tokens"]
                        if wait_time_ms > 0:
                            tokens_per_second = tokens / (wait_time_ms / 1000)

            if self.logger:
                metadata = {
                    "temperature": temperature,
                    "prompt_type": prompt_type,
                    "timing": {
                        "wait_time_ms": wait_time_ms,
                        "tokens": tokens,
                        "tokens_per_second": tokens_per_second,
                    },
                }
                self.logger(self.model, prompt, result, metadata)

            return result
        except Exception as e:
            raise LLMError(f"OpenAI API error: {e}") from e
