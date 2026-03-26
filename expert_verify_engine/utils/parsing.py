import json
import re
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class SchemaValidationError(Exception):
    pass


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            return match.group(1)
    if text.startswith("{"):
        return text
    raise ValueError("No JSON found in text")


def parse_json(text: str, model: type[T]) -> T:  # noqa: UP047
    try:
        json_str = extract_json(text)
        data = json.loads(json_str)
        return model.model_validate(data)
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON: {e}") from e
    except Exception as e:
        raise SchemaValidationError(f"Validation failed: {e}") from e
