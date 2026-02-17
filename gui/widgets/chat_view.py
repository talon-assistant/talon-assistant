from datetime import datetime
from PyQt6.QtWidgets import (QScrollArea, QWidget, QVBoxLayout, QLabel,
                             QFrame, QTextBrowser, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextDocument


class ChatBubble(QFrame):
    """Single message bubble. Styled via QSS objectName.

    Uses QTextBrowser for the message body so that long text wraps properly
    and the bubble grows in height to fit all content.
    """

    def __init__(self, text, role, parent=None):
        super().__init__(parent)
        self.setObjectName(f"chat_bubble_{role}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Role label
        role_names = {"user": "You", "assistant": "Talon",
                      "error": "Error", "system": "System"}
        role_label = QLabel(role_names.get(role, role))
        role_label.setObjectName("bubble_role")
        role_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        layout.addWidget(role_label)

        # Message body — QTextBrowser handles word-wrap + dynamic height
        self.msg_view = QTextBrowser()
        self.msg_view.setObjectName("bubble_text")
        self.msg_view.setPlainText(text)
        self.msg_view.setReadOnly(True)
        self.msg_view.setOpenExternalLinks(False)
        self.msg_view.setFrameShape(QFrame.Shape.NoFrame)
        self.msg_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.msg_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Transparent background — the bubble frame provides the colour
        self.msg_view.setStyleSheet(
            "QTextBrowser { background: transparent; border: none; }")

        # Calculate the exact height needed for the text content
        self.msg_view.document().setDocumentMargin(0)
        self.msg_view.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Fixed)
        layout.addWidget(self.msg_view)

        # Store text for deferred height calculation
        self._text = text

    def resizeEvent(self, event):
        """Recalculate text height when bubble width changes."""
        super().resizeEvent(event)
        self._adjust_height()

    def showEvent(self, event):
        """Calculate initial height when first shown."""
        super().showEvent(event)
        self._adjust_height()

    def _adjust_height(self):
        """Set the QTextBrowser height to exactly fit the text content."""
        doc = self.msg_view.document()
        # Tell the document what width it has available
        content_width = self.msg_view.viewport().width()
        if content_width > 0:
            doc.setTextWidth(content_width)
        doc_height = int(doc.size().height()) + 4  # small padding
        self.msg_view.setFixedHeight(max(20, doc_height))


class ChatView(QScrollArea):
    """Scrollable chat history with auto-scroll to bottom."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._messages = []  # list of dicts: {role, text, timestamp}

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.addStretch()
        self.setWidget(container)

    def _add_message(self, text, role):
        """Internal: add bubble and track message."""
        self._messages.append({
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat(timespec='seconds'),
        })
        bubble = ChatBubble(text, role)
        self._layout.insertWidget(self._layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def add_user_message(self, text):
        """Add a user message bubble (right-aligned via QSS margin-left)."""
        self._add_message(text, "user")

    def add_assistant_message(self, text):
        """Add an assistant response bubble (left-aligned via QSS margin-right)."""
        self._add_message(text, "assistant")

    def add_error_message(self, text):
        """Add an error message bubble."""
        self._add_message(f"Error: {text}", "error")

    def add_system_message(self, text):
        """Add a system/info message bubble (centered via QSS margins)."""
        self._add_message(text, "system")

    def get_messages(self):
        """Return list of message dicts for saving."""
        return list(self._messages)

    def clear_messages(self):
        """Remove all bubbles and reset message tracking."""
        self._messages.clear()
        # Remove all widgets except the stretch at the end
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def load_messages(self, messages):
        """Clear and re-populate from a list of ChatMessage or dicts."""
        self.clear_messages()
        for msg in messages:
            # Support both ChatMessage objects and dicts
            if hasattr(msg, 'role'):
                role, text, ts = msg.role, msg.text, msg.timestamp
            else:
                role = msg.get("role", "system")
                text = msg.get("text", "")
                ts = msg.get("timestamp", "")

            self._messages.append({
                "role": role,
                "text": text,
                "timestamp": ts,
            })
            bubble = ChatBubble(text, role)
            self._layout.insertWidget(self._layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Scroll to bottom after layout updates."""
        QTimer.singleShot(50, self._do_scroll)

    def _do_scroll(self):
        """Actual scroll — safe to call even during shutdown."""
        try:
            sb = self.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())
        except RuntimeError:
            pass
