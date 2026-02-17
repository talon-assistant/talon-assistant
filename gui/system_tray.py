from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QMetaObject, Qt


def _create_default_icon():
    """Create a simple 64x64 blue circle 'J' icon programmatically."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#89b4fa"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, 60, 60)
    painter.setPen(QColor("#1e1e2e"))
    painter.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "J")
    painter.end()
    return QIcon(pixmap)


class SystemTrayManager(QObject):
    """Manages system tray icon, context menu, notifications, and global hotkeys."""

    show_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    exit_requested = pyqtSignal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self._window = window
        self._setup_tray_icon()
        self._setup_hotkey()

    def _setup_tray_icon(self):
        """Create QSystemTrayIcon with context menu."""
        self.tray_icon = QSystemTrayIcon(self._window)
        self.tray_icon.setIcon(_create_default_icon())
        self.tray_icon.setToolTip("Talon Assistant")

        menu = QMenu()
        show_action = QAction("Show Talon", menu)
        show_action.triggered.connect(self._show_window)
        menu.addAction(show_action)

        hide_action = QAction("Hide to Tray", menu)
        hide_action.triggered.connect(self._hide_window)
        menu.addAction(hide_action)

        menu.addSeparator()

        exit_action = QAction("Exit", menu)
        exit_action.triggered.connect(self.exit_requested.emit)
        menu.addAction(exit_action)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        """Double-click on tray icon shows/hides window."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self._window.isVisible():
                self._hide_window()
            else:
                self._show_window()

    def _show_window(self):
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()
        self.show_requested.emit()

    def _hide_window(self):
        self._window.hide()
        self.hide_requested.emit()

    def _setup_hotkey(self):
        """Register global hotkey using pynput (Ctrl+Shift+J)."""
        self._listener = None
        try:
            from pynput import keyboard

            COMBO = {keyboard.Key.ctrl_l, keyboard.Key.shift, keyboard.KeyCode.from_char('j')}
            self._current_keys = set()

            def on_press(key):
                self._current_keys.add(key)
                if all(k in self._current_keys for k in COMBO):
                    # Cross to GUI thread safely
                    QMetaObject.invokeMethod(
                        self, "_toggle_window",
                        Qt.ConnectionType.QueuedConnection)

            def on_release(key):
                self._current_keys.discard(key)

            self._listener = keyboard.Listener(
                on_press=on_press, on_release=on_release)
            self._listener.daemon = True
            self._listener.start()
        except ImportError:
            print("[SystemTray] pynput not installed; global hotkey disabled.")
        except Exception as e:
            print(f"[SystemTray] Global hotkey setup failed: {e}")

    @pyqtSlot()
    def _toggle_window(self):
        if self._window.isVisible():
            self._hide_window()
        else:
            self._show_window()

    def show_notification(self, title, message, duration_ms=5000):
        """Show a desktop notification via the system tray icon."""
        if self.tray_icon.supportsMessages():
            self.tray_icon.showMessage(
                title, message,
                QSystemTrayIcon.MessageIcon.Information,
                duration_ms)

    def cleanup(self):
        """Stop hotkey listener, hide tray icon."""
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
        self.tray_icon.hide()
