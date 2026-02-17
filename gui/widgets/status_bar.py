from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel


class StatusBarWidget(QWidget):
    """Custom status bar content with colored status indicators."""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)

        # LLM status
        self.llm_indicator = QLabel("LLM: Checking...")
        self.llm_indicator.setObjectName("status_llm_unknown")
        layout.addWidget(self.llm_indicator)

        layout.addWidget(self._separator())

        # Voice status
        self.voice_indicator = QLabel("Voice: Off")
        self.voice_indicator.setObjectName("status_voice_off")
        layout.addWidget(self.voice_indicator)

        layout.addWidget(self._separator())

        # Activity
        self.activity_label = QLabel("Activity: Idle")
        self.activity_label.setObjectName("status_idle")
        layout.addWidget(self.activity_label)

        layout.addStretch()

    def _separator(self):
        sep = QLabel("|")
        sep.setObjectName("status_separator")
        return sep

    def set_llm_status(self, connected):
        text = "LLM: Connected" if connected else "LLM: Disconnected"
        obj_name = "status_llm_connected" if connected else "status_llm_disconnected"
        self.llm_indicator.setText(text)
        self.llm_indicator.setObjectName(obj_name)
        self.llm_indicator.style().unpolish(self.llm_indicator)
        self.llm_indicator.style().polish(self.llm_indicator)

    def set_voice_status(self, status):
        self.voice_indicator.setText(f"Voice: {status.capitalize()}")
        obj_name = f"status_voice_{status}"
        self.voice_indicator.setObjectName(obj_name)
        self.voice_indicator.style().unpolish(self.voice_indicator)
        self.voice_indicator.style().polish(self.voice_indicator)

    def set_activity(self, activity):
        self.activity_label.setText(f"Activity: {activity.capitalize()}")
        obj_name = f"status_{activity}"
        self.activity_label.setObjectName(obj_name)
        self.activity_label.style().unpolish(self.activity_label)
        self.activity_label.style().polish(self.activity_label)
