from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt


class TextInput(QWidget):
    """Text input bar with send button and command history."""

    command_submitted = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._history = []
        self._history_index = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a command...")
        self.input_field.returnPressed.connect(self._submit)
        self.input_field.installEventFilter(self)
        layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.setFixedWidth(70)
        self.send_button.clicked.connect(self._submit)
        layout.addWidget(self.send_button)

    def _submit(self):
        text = self.input_field.text().strip()
        if text:
            self._history.append(text)
            self._history_index = len(self._history)
            self.command_submitted.emit(text)
            self.input_field.clear()

    def set_enabled(self, enabled):
        """Enable/disable input during command processing."""
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        if enabled:
            self.input_field.setFocus()

    def eventFilter(self, obj, event):
        """Handle Up/Down arrow keys for command history."""
        if obj == self.input_field and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Up and self._history:
                self._history_index = max(0, self._history_index - 1)
                self.input_field.setText(self._history[self._history_index])
                return True
            elif event.key() == Qt.Key.Key_Down and self._history:
                self._history_index = min(len(self._history), self._history_index + 1)
                if self._history_index < len(self._history):
                    self.input_field.setText(self._history[self._history_index])
                else:
                    self.input_field.clear()
                return True
        return super().eventFilter(obj, event)
