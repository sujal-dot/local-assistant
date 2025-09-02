# extras.py - helper classes for GUI animations and voice input
import threading
import json
import os
import time

from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QLabel, QApplication

try:
    from vosk import Model, KaldiRecognizer
    import pyaudio
except ImportError:
    print("Vosk or PyAudio not installed. Please run: pip install vosk PyAudio")
    exit()

class LoadingAnimation(QLabel):
    """Simple loading animation dots"""
    def __init__(self, text="Loading", parent=None):
        super().__init__(text, parent)
        self.base_text = text
        self.counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dots)

    def start(self):
        self.timer.start(300)

    def stop(self):
        self.timer.stop()
        self.setText(self.base_text)

    def update_dots(self):
        self.counter = (self.counter + 1) % 4
        self.setText(self.base_text + "." * self.counter)

class ListeningGlow:
    """Glow effect for the microphone button during listening"""
    def __init__(self, widget):
        self.widget = widget
        self.original_palette = widget.palette()
        self.color_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.glow)

    def start(self):
        self.timer.start(100)

    def stop(self):
        self.timer.stop()
        self.widget.setPalette(self.original_palette)

    def glow(self):
        colors = [QColor("cyan"), QColor("magenta"), QColor("white")]
        palette = self.widget.palette()
        palette.setColor(QPalette.Button, colors[self.color_index])
        self.widget.setPalette(palette)
        self.color_index = (self.color_index + 1) % len(colors)

class VoskWorker(QThread):
    """
    Worker thread for Vosk speech recognition.
    Performs the heavy lifting so the GUI doesn't freeze.
    """
    partial_result = pyqtSignal(str)
    final_result = pyqtSignal(str)
    voice_error = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self._is_listening = False
        
        # Audio stream parameters
        self.CHUNK = 8192
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

    def run(self):
        """Main recognition loop"""
        if not os.path.exists(self.model_path):
            self.voice_error.emit(f"Vosk model not found at '{self.model_path}'.")
            return
            
        try:
            model = Model(self.model_path)
            rec = KaldiRecognizer(model, self.RATE)
            p = pyaudio.PyAudio()
            stream = p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK)

            self._is_listening = True
            print("Listening for voice input...")
            
            while self._is_listening:
                data = stream.read(self.CHUNK)
                
                # Check for a pause (final result)
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get('text', '')
                    if text:
                        self.final_result.emit(text)
                else:
                    # Get partial result
                    partial = json.loads(rec.PartialResult())
                    text = partial.get('partial', '')
                    if text:
                        self.partial_result.emit(text)
                        
        except Exception as e:
            self.voice_error.emit(f"Voice recognition error: {e}")
        finally:
            self._is_listening = False
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()

    def stop(self):
        """Stops the recognition loop gracefully"""
        self._is_listening = False

class VoiceInputHandler:
    """
    Manages the Vosk thread and glow effect, connecting them
    to the main application's UI.
    """
    def __init__(self, parent_widget, model_path, on_partial_text, on_final_text):
        self.parent_widget = parent_widget
        self.model_path = model_path
        self.on_partial_text = on_partial_text
        self.on_final_text = on_final_text
        self.vosk_worker = None
        self.mic_glow = ListeningGlow(parent_widget)
        
    def start(self):
        """Starts the voice recognition process and glow effect"""
        if self.vosk_worker and self.vosk_worker.isRunning():
            print("Already listening.")
            return

        self.mic_glow.start()
        self.vosk_worker = VoskWorker(self.model_path)
        self.vosk_worker.partial_result.connect(self.on_partial_text)
        self.vosk_worker.final_result.connect(self.handle_final_result)
        self.vosk_worker.voice_error.connect(self.handle_error)
        self.vosk_worker.start()
        
    def stop(self):
        """Stops the voice recognition and glow effect"""
        if self.vosk_worker:
            self.vosk_worker.stop()
            self.vosk_worker.wait() # Wait for the thread to finish
        self.mic_glow.stop()
        
    def handle_final_result(self, text):
        """Handles the final result and triggers the auto-send"""
        self.on_final_text(text)
        self.stop()
        
    def handle_error(self, error_message):
        """Handles any errors from the voice recognizer"""
        print(f"Error: {error_message}")
        self.on_final_text(f"[Voice error: {error_message}]")
        self.stop()

# Example usage (for demonstration)
if __name__ == '__main__':
    class MyWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Vosk Demo")
            self.setGeometry(100, 100, 400, 300)
            
            # Simple UI elements
            self.layout = QVBoxLayout()
            self.status_label = QLabel("Click the button to start voice input.")
            self.input_box = QLineEdit()
            self.mic_button = QPushButton("🎤 Start Listening")
            
            self.layout.addWidget(self.status_label)
            self.layout.addWidget(self.input_box)
            self.layout.addWidget(self.mic_button)
            self.setLayout(self.layout)
            
            self.mic_button.clicked.connect(self.toggle_listening)

            # Initialize the voice handler with dummy callback functions
            self.voice_handler = VoiceInputHandler(
                parent_widget=self.mic_button, 
                model_path="vosk-model-en-us-0.22", # Update this path to your model folder
                on_partial_text=self.update_input_box,
                on_final_text=self.send_message
            )
            self.is_listening = False

        def toggle_listening(self):
            if self.is_listening:
                self.voice_handler.stop()
                self.mic_button.setText("🎤 Start Listening")
                self.is_listening = False
            else:
                self.voice_handler.start()
                self.mic_button.setText("🔴 Listening...")
                self.is_listening = True

        def update_input_box(self, text):
            self.input_box.setText(text)
            
        def send_message(self, text):
            # This simulates what happens when the final text is "sent"
            self.status_label.setText(f"Message Sent: {text}")
            self.input_box.setText("")
            self.mic_button.setText("🎤 Start Listening")
            self.is_listening = False
    
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()