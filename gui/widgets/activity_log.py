from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
from datetime import datetime


class ActivityLog(QWidget):
    """Collapsible log panel showing internal activity with timestamps."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar with collapse toggle
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 2, 8, 2)

        self.toggle_button = QPushButton("Activity Log  [-]")
        self.toggle_button.setObjectName("log_toggle")
        self.toggle_button.setFlat(True)
        self.toggle_button.clicked.connect(self._toggle)
        header_layout.addWidget(self.toggle_button)

        header_layout.addStretch()

        clear_button = QPushButton("Clear")
        clear_button.setFixedWidth(60)
        clear_button.clicked.connect(self._clear)
        header_layout.addWidget(clear_button)

        layout.addWidget(header)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setObjectName("activity_log_text")
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

        self._collapsed = False

    def append_entry(self, source, message):
        """Add a timestamped, color-coded log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "Routing": "#4fc3f7",
            "Talent": "#81c784",
            "LLM": "#ffb74d",
            "TTS": "#ce93d8",
            "Voice": "#4dd0e1",
            "Error": "#ef5350",
            "System": "#90a4ae",
        }
        color = color_map.get(source, "#b0bec5")
        html = (f'<span style="color:#666">{timestamp}</span> '
                f'<span style="color:{color}">[{source}]</span> '
                f'<span style="color:#ddd">{message}</span>')
        self.log_text.append(html)

    def append_raw(self, text):
        """Append raw print() output from OutputInterceptor."""
        text = text.strip()
        if not text:
            return

        # Skip lines that go to the chat view
        if text.startswith("Talon:"):
            return
        # Skip decorative separators
        if text.startswith("=") or text.startswith("COMMAND:"):
            return

        # Classify by content
        if "[Routing]" in text:
            self.append_entry("Routing", text)
        elif "[Talents]" in text or "[Loading]" in text:
            self.append_entry("Talent", text)
        elif "[Memory]" in text:
            self.append_entry("System", text)
        elif "Error" in text or "error" in text.lower():
            self.append_entry("Error", text)
        elif "TTS" in text or "Speaking" in text:
            self.append_entry("TTS", text)
        elif "Whisper" in text or "Voice" in text or "Listening" in text:
            self.append_entry("Voice", text)
        else:
            self.append_entry("System", text)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self.log_text.setVisible(not self._collapsed)
        symbol = "+" if self._collapsed else "-"
        self.toggle_button.setText(f"Activity Log  [{symbol}]")

    def _clear(self):
        self.log_text.clear()
