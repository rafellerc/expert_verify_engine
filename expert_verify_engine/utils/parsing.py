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


def sanitize_json_string(text: str) -> str:
    result = []
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
            result.append(char)
        elif in_string:
            if char == "\n" or char == "\r":
                result.append("\\n")
            elif char == "\t":
                result.append("\\t")
            elif ord(char) >= 32:
                result.append(char)
            else:
                pass
        else:
            if ord(char) >= 32 or char in "{}[]:,":
                result.append(char)
        i += 1
    return "".join(result)


def parse_json(text: str, model: type[T]) -> T:  # noqa: UP047
    try:
        json_str = extract_json(text)
        json_str = sanitize_json_string(json_str)
        brace_count = 0
        start_idx = None
        for i, char in enumerate(json_str):
            if char == "{":
                if start_idx is None:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    json_str = json_str[start_idx : i + 1]
                    break
        data = json.loads(json_str)
        return model.model_validate(data)
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON: {e}") from e
    except Exception as e:
        raise SchemaValidationError(f"Validation failed: {e}") from e
