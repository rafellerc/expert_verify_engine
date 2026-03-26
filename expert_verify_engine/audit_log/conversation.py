from pathlib import Path


class ConversationLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def add(self, role: str, message: str) -> None:
        self.messages.append((role, message))

    def get_history(self) -> str:
        lines = []
        for role, msg in self.messages:
            lines.append(f"{role}: {msg}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.get_history())

    def load(self, path: Path) -> None:
        content = path.read_text()
        self.messages = []
        for line in content.split("\n"):
            if ": " in line:
                role, msg = line.split(": ", 1)
                self.messages.append((role, msg))

    def __str__(self) -> str:
        return self.get_history()
