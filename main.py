import sys
import os
import signal
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows DLL fix
if sys.platform == 'win32':
    os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))


def _install_exception_hook():
    """Install a global exception hook so unhandled exceptions in Qt slots
    produce a traceback instead of a silent crash / segfault."""
    import faulthandler
    faulthandler.enable()

    _original_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        sys.stderr.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        sys.stderr.flush()
        _original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def main():
    mode = "gui"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "gui":
        _install_exception_hook()

        # ── Step 1: Build TalonAssistant BEFORE QApplication ──────────
        # CTranslate2 (used by faster-whisper) segfaults when WhisperModel
        # is created after QApplication has initialised its platform plugins.
        # Loading all models first avoids the conflict entirely.
        print("Loading Talon (models + services)...")
        print("  This may take 10-30 seconds on first launch.\n")
        assistant = None
        init_error = None
        try:
            from core.assistant import TalonAssistant
            assistant = TalonAssistant(config_dir="config")
        except Exception as e:
            init_error = str(e)
            traceback.print_exc()

        # ── Step 2: Now it's safe to create Qt ─────────────────────────
        from PyQt6.QtWidgets import QApplication
        from gui.main_window import MainWindow
        from gui.assistant_bridge import AssistantBridge
        from gui.output_interceptor import OutputInterceptor

        app = QApplication(sys.argv)

        # Allow Ctrl+C to kill the app from the terminal
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Theme manager (replaces inline QSS loading)
        from gui.theme_manager import ThemeManager
        theme_manager = ThemeManager(app, config_dir="config")

        # Output interceptor
        interceptor = OutputInterceptor(sys.stdout)
        sys.stdout = interceptor

        # Create window
        bridge = AssistantBridge(config_dir="config")
        window = MainWindow(bridge, theme_manager=theme_manager,
                            config_dir="config")

        # System tray
        from gui.system_tray import SystemTrayManager
        tray_manager = SystemTrayManager(window)
        window.system_tray = tray_manager
        tray_manager.exit_requested.connect(window._force_exit)

        interceptor.text_written.connect(window.activity_log.append_raw)

        window.show()

        # Hand the pre-built assistant to the bridge (or show error)
        if assistant is not None:
            bridge.set_assistant(assistant)
        else:
            window.chat_view.add_error_message(
                f"Initialization failed: {init_error}"
            )
            window.text_input.set_enabled(False)

        exit_code = app.exec()

        # Clean up stdout interceptor
        sys.stdout = interceptor._original or sys.__stdout__

        # Force-kill the process to ensure all threads (pynput, sounddevice, etc.) die
        # QThread cleanup already attempted in closeEvent -> bridge.cleanup()
        os._exit(exit_code)

    elif mode in ("voice", "text", "both"):
        from core.assistant import TalonAssistant
        talon = TalonAssistant(config_dir="config")
        talon.run(mode=mode)

    else:
        print(f"Unknown mode: {mode}. Use gui, voice, text, or both.")
        sys.exit(1)


if __name__ == "__main__":
    main()
