import threading
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class InitWorker(QThread):
    """Initializes TalonAssistant off the GUI thread.

    TalonAssistant.__init__() loads Whisper, ChromaDB, SentenceTransformer,
    connects to Hue, tests LLM — total 10-30 seconds.
    """

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, config_dir="config"):
        super().__init__()
        self.config_dir = config_dir

    def run(self):
        try:
            from core.assistant import TalonAssistant
            assistant = TalonAssistant(config_dir=self.config_dir)
            self.finished.emit(assistant)
        except Exception as e:
            self.error.emit(str(e))


class CommandWorker(QThread):
    """Runs process_command() off the GUI thread.

    Calls with speak_response=False so TTS is handled separately by the bridge.
    """

    response_ready = pyqtSignal(str, str, str, bool)  # command, response, talent_name, success
    error = pyqtSignal(str, str)  # command, error_message

    def __init__(self, assistant, command):
        super().__init__()
        self.assistant = assistant
        self.command = command

    def run(self):
        try:
            result = self.assistant.process_command(self.command, speak_response=False)
            if result and isinstance(result, dict):
                self.response_ready.emit(
                    self.command,
                    result.get("response", ""),
                    result.get("talent", ""),
                    result.get("success", True)
                )
            elif result is None:
                # Command was too short / empty
                self.response_ready.emit(
                    self.command, "Command too short to process.", "", False
                )
            else:
                # Unexpected return type
                self.response_ready.emit(self.command, str(result), "", True)
        except Exception as e:
            self.error.emit(self.command, str(e))


class TTSWorker(QThread):
    """Runs text-to-speech off the GUI thread.

    voice.speak() uses asyncio.run() internally, which is safe from a QThread.
    Supports mid-speech interruption via request_stop().
    """

    started_speaking = pyqtSignal()
    finished_speaking = pyqtSignal()
    stopped_early = pyqtSignal()  # emitted when TTS was interrupted
    error = pyqtSignal(str)

    def __init__(self, voice_system, text):
        super().__init__()
        self.voice = voice_system
        self.text = text

    def run(self):
        try:
            self.started_speaking.emit()
            completed = self.voice.speak(self.text)
            if completed:
                self.finished_speaking.emit()
            else:
                self.stopped_early.emit()
        except Exception as e:
            self.error.emit(str(e))

    def request_stop(self):
        """Ask the voice system to stop playback immediately."""
        self.voice.stop_speaking()


class VoiceListenWorker(QThread):
    """Controllable wake word listener that reimplements the loop from VoiceSystem.

    Does NOT call voice.listen_for_wake_word() because that has no stop mechanism.
    Instead, reimplements the loop body with a thread-safe _running flag.
    """

    wake_word_detected = pyqtSignal()
    command_transcribed = pyqtSignal(str)
    audio_level = pyqtSignal(float)
    heard_text = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, voice_system):
        super().__init__()
        self.voice = voice_system
        self._lock = threading.Lock()
        self._running = False

    @property
    def running(self):
        with self._lock:
            return self._running

    @running.setter
    def running(self, value):
        with self._lock:
            self._running = value

    def run(self):
        self.running = True
        self.status_changed.emit("listening")

        while self.running:
            try:
                # Record audio chunk (blocking for chunk_duration seconds)
                audio = self.voice.record_audio(self.voice.chunk_duration)

                if not self.running:
                    break

                # Check energy/variance thresholds
                audio_energy = float(np.abs(audio).mean())
                audio_variance = float(np.var(audio))

                # Emit normalized audio level (0.0-1.0)
                normalized_level = min(1.0, audio_energy / 500.0)
                self.audio_level.emit(normalized_level)

                if audio_energy < self.voice.energy_threshold:
                    continue
                if audio_variance < self.voice.variance_threshold:
                    continue

                # Transcribe chunk
                self.status_changed.emit("transcribing")
                text = self.voice.transcribe_audio(audio)

                if text.strip() in self.voice.noise_words:
                    self.status_changed.emit("listening")
                    continue

                if text and len(text) > 1:
                    self.heard_text.emit(text)

                # Check for wake word
                if any(ww in text for ww in self.voice.wake_words):
                    self.wake_word_detected.emit()
                    self.status_changed.emit("recording_command")

                    # Acknowledge wake word — speak "Ready" so user knows to give command
                    try:
                        self.voice.speak("Ready")
                    except Exception:
                        pass

                    # Record command audio
                    command_audio = self.voice.record_audio(self.voice.command_duration)

                    if not self.running:
                        break

                    self.status_changed.emit("transcribing")
                    command_text = self.voice.transcribe_audio(command_audio)

                    if command_text and len(command_text) > 1:
                        self.command_transcribed.emit(command_text)

                self.status_changed.emit("listening")

            except Exception as e:
                if not self.running:
                    break
                self.error.emit(str(e))

    def stop(self):
        """Stop the listening loop. May take up to chunk_duration to finish."""
        self.running = False
