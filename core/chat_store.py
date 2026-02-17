import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ChatMessage:
    """Single chat message."""
    role: str       # "user", "assistant", "system", "error"
    text: str
    timestamp: str  # ISO 8601

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(
            role=d.get("role", "system"),
            text=d.get("text", ""),
            timestamp=d.get("timestamp", ""),
        )


class ChatStore:
    """Manages saving/loading conversation sessions as JSON files.

    Conversations are stored in data/conversations/ as individual JSON files.
    """

    CONVERSATIONS_DIR = "data/conversations"

    def __init__(self):
        os.makedirs(self.CONVERSATIONS_DIR, exist_ok=True)

    def save_conversation(self, messages, filepath=None):
        """Save messages to JSON file. Returns the filepath used."""
        if not messages:
            return None

        if filepath is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filepath = os.path.join(
                self.CONVERSATIONS_DIR, f"conversation_{ts}.json")

        data = {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": [m.to_dict() if isinstance(m, ChatMessage)
                         else m for m in messages],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath

    def load_conversation(self, filepath):
        """Load messages from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [ChatMessage.from_dict(m) for m in data.get("messages", [])]

    def list_conversations(self):
        """Returns list of conversation metadata sorted by most recent first."""
        conversations = []
        for filename in os.listdir(self.CONVERSATIONS_DIR):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(self.CONVERSATIONS_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                messages = data.get("messages", [])
                preview = ""
                for m in messages:
                    if m.get("role") == "user":
                        preview = m.get("text", "")[:80]
                        break
                conversations.append({
                    "filepath": filepath,
                    "filename": filename,
                    "timestamp": data.get("timestamp", ""),
                    "message_count": data.get("message_count", len(messages)),
                    "preview": preview,
                })
            except (json.JSONDecodeError, OSError):
                continue

        conversations.sort(key=lambda c: c["timestamp"], reverse=True)
        return conversations

    def export_as_text(self, messages, filepath):
        """Export conversation as human-readable plain text."""
        lines = []
        for m in messages:
            msg = m if isinstance(m, ChatMessage) else ChatMessage.from_dict(m)
            role_name = {"user": "You", "assistant": "Talon",
                         "error": "Error", "system": "System"}.get(
                msg.role, msg.role)
            lines.append(f"[{msg.timestamp}] {role_name}:")
            lines.append(msg.text)
            lines.append("")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

    def export_as_markdown(self, messages, filepath):
        """Export conversation as Markdown with role headers."""
        lines = ["# Talon Conversation", ""]
        for m in messages:
            msg = m if isinstance(m, ChatMessage) else ChatMessage.from_dict(m)
            role_name = {"user": "You", "assistant": "Talon",
                         "error": "Error", "system": "System"}.get(
                msg.role, msg.role)
            lines.append(f"### {role_name}")
            lines.append(f"*{msg.timestamp}*")
            lines.append("")
            lines.append(msg.text)
            lines.append("")
            lines.append("---")
            lines.append("")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

    def delete_conversation(self, filepath):
        """Delete a saved conversation file."""
        if os.path.exists(filepath):
            os.remove(filepath)
