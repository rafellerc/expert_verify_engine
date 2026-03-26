from openai import OpenAI

from expert_verify_engine.app.config import get_config
from expert_verify_engine.utils.retry import default_retry


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self) -> None:
        self.api_key = get_config("api_key")
        self.model = get_config("model")
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY not set")

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    @default_retry
    def chat(self, prompt: str, temperature: float | None = None) -> str:
        if temperature is None:
            temperature = get_config("temperature")

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI API error: {e}") from e
