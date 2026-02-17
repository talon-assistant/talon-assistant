from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
                             QListWidget, QListWidgetItem, QTextBrowser,
                             QPushButton, QLabel, QMessageBox)
from PyQt6.QtCore import Qt


class ChatHistoryDialog(QDialog):
    """Browse saved conversations, preview them, and load."""

    def __init__(self, chat_store, parent=None):
        super().__init__(parent)
        self.chat_store = chat_store
        self.selected_filepath = None
        self.setWindowTitle("Conversation History")
        self.setMinimumSize(700, 450)
        self.resize(800, 500)

        layout = QVBoxLayout(self)

        # Splitter: list | preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: conversation list
        left = QListWidget()
        left.setObjectName("chat_history_list")
        self.list_widget = left
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        splitter.addWidget(left)

        # Right: preview pane
        self.preview = QTextBrowser()
        self.preview.setObjectName("chat_preview_pane")
        self.preview.setReadOnly(True)
        splitter.addWidget(self.preview)

        splitter.setSizes([300, 500])
        layout.addWidget(splitter)

        # Buttons
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._on_delete)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(delete_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._populate_list()

    def _populate_list(self):
        self.list_widget.clear()
        conversations = self.chat_store.list_conversations()
        for conv in conversations:
            label = f"{conv['timestamp'][:19]}  ({conv['message_count']} msgs)"
            if conv.get("preview"):
                label += f"\n  {conv['preview']}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, conv["filepath"])
            self.list_widget.addItem(item)

    def _on_selection_changed(self, current, previous):
        if current is None:
            self.preview.clear()
            return
        filepath = current.data(Qt.ItemDataRole.UserRole)
        try:
            messages = self.chat_store.load_conversation(filepath)
            lines = []
            for msg in messages:
                role_name = {"user": "You", "assistant": "Talon",
                             "error": "Error", "system": "System"}.get(
                    msg.role, msg.role)
                lines.append(f"<b>{role_name}</b> "
                             f"<span style='color: #6c7086; font-size: 11px;'>"
                             f"{msg.timestamp}</span><br>{msg.text}<br><br>")
            self.preview.setHtml("".join(lines))
        except Exception as e:
            self.preview.setPlainText(f"Error loading: {e}")

    def _on_load(self):
        current = self.list_widget.currentItem()
        if current is None:
            return
        self.selected_filepath = current.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_delete(self):
        current = self.list_widget.currentItem()
        if current is None:
            return
        filepath = current.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, "Delete Conversation",
            "Delete this saved conversation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.chat_store.delete_conversation(filepath)
            self._populate_list()
            self.preview.clear()
