# frontend/gui.py
"""
PyQt6 GUI with:
- history tab (left)
- streaming assistant output (appears as typing)
- copy button for AI outputs
- mic button for voice input (uses offline VOSK STT from backend/speech.py)
- Enter to send (Shift+Enter = newline)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout,
    QScrollArea, QFrame, QListWidget, QListWidgetItem, QSplitter, QToolButton
)
from PyQt6.QtGui import QPixmap, QGuiApplication
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# backend
from backend import server, speech   # ✅ now use backend.speech for STT

# --- Worker threads -----------------------------------------------------------------
class StreamWorker(QThread):
    partial = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, prompt, max_tokens=256, temperature=0.2):
        super().__init__()
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self):
        try:
            collected = ""
            for chunk in server.ask_stream(self.prompt, max_tokens=self.max_tokens, temperature=self.temperature):
                self.partial.emit(chunk)
                collected += chunk
            self.finished.emit(collected)
        except Exception as e:
            self.finished.emit(f"[error] {e}")


class STTWorker(QThread):
    result = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, device_index=None):
        super().__init__()
        self.device_index = device_index

    def run(self):
        try:
            # init vosk model (fixed path inside backend/speech.py)
            speech.init_stt()

            def callback(text, final):
                if final:
                    self.result.emit(text)

            speech.start_listening(callback, device_index=self.device_index)
        except Exception as e:
            self.error.emit(str(e))


# --- GUI ----------------------------------------------------------------------------
class ChatWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        try:
            server.init_model()
        except Exception as e:
            print("[gui] server.init_model error:", e)
        self.history = server.load_history()
        self.populate_history_list()

    def init_ui(self):
        self.setWindowTitle("My Assistant")
        self.resize(1200, 780)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # left history
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_click)
        left_frame = QFrame()
        left_layout = QVBoxLayout()
        title = QLabel("<b>History</b>")
        left_layout.addWidget(title)
        left_layout.addWidget(self.history_list)
        left_frame.setLayout(left_layout)
        splitter.addWidget(left_frame)
        splitter.setStretchFactor(0, 1)

        # right chat
        right_frame = QFrame()
        right_layout = QVBoxLayout()

        header_h = QHBoxLayout()
        logo = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.png')
        if os.path.exists(logo_path):
            logo.setPixmap(QPixmap(logo_path).scaledToHeight(44, Qt.TransformationMode.SmoothTransformation))
        header_h.addWidget(logo)
        self.status_label = QLabel("Model: searching...")
        header_h.addWidget(self.status_label)
        header_h.addStretch()
        right_layout.addLayout(header_h)

        self.chat_area = QVBoxLayout()
        chat_frame = QFrame()
        chat_frame.setLayout(self.chat_area)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(chat_frame)
        scroll.setMinimumHeight(420)
        right_layout.addWidget(scroll)

        # input + mic + send
        input_h = QHBoxLayout()
        self.input = QTextEdit()
        self.input.setPlaceholderText("Type your message. Enter = send, Shift+Enter = newline.")
        self.input.setFixedHeight(110)
        self.input.installEventFilter(self)

        self.mic_btn = QToolButton()
        self.mic_btn.setText("🎤")
        self.mic_btn.clicked.connect(self.on_mic)
        self.mic_btn.setToolTip("Voice input (offline VOSK)")
        input_h.addWidget(self.input)
        input_h.addWidget(self.mic_btn)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.on_send)
        input_h.addWidget(send_btn)
        right_layout.addLayout(input_h)

        right_frame.setLayout(right_layout)
        splitter.addWidget(right_frame)
        splitter.setStretchFactor(1, 3)

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        if getattr(server, "_model_path", None):
            self.status_label.setText(f"Model: {server._model_path}")
        else:
            mp = server.find_model_file()
            if mp:
                self.status_label.setText(f"Model: {mp}")
            else:
                self.status_label.setText("Model: not found (place .gguf on Desktop or repo/model/)")

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if obj is self.input and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self.on_send()
                return True
        return super().eventFilter(obj, event)

    def append_user(self, text):
        lbl = QLabel(f"<b>You</b><br/>{text}")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("background: rgba(40,40,40,0.6); padding:8px; margin:6px; border-radius:8px;")
        self.chat_area.addWidget(lbl)

    def append_assistant_streaming(self, initial_text=""):
        container = QFrame()
        v = QVBoxLayout()
        lbl = QLabel()
        lbl.setWordWrap(True)
        lbl.setText(f"<b>Assistant</b><br/>{initial_text}")
        lbl.setStyleSheet("background: rgba(25,35,60,0.6); padding:8px; margin:6px; border-radius:8px;")
        v.addWidget(lbl)
        copy_btn = QPushButton("Copy")
        copy_btn.setMaximumWidth(80)
        v.addWidget(copy_btn, alignment=Qt.AlignmentFlag.AlignRight)
        container.setLayout(v)
        self.chat_area.addWidget(container)
        return lbl, copy_btn

    def on_send(self):
        prompt = self.input.toPlainText().strip()
        if not prompt:
            return
        self.append_user(prompt)
        self.input.clear()

        lbl, copy_btn = self.append_assistant_streaming("")
        self.worker = StreamWorker(prompt, max_tokens=400, temperature=0.1)
        buffer = {"text": ""}

        def on_partial(chunk):
            buffer["text"] += chunk
            lbl.setText(f"<b>Assistant</b><br/>{buffer['text']}")

        def on_finished(final):
            lbl.setText(f"<b>Assistant</b><br/>{final}")
            copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(final))
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": final}]
            title = final.strip().split("\n", 1)[0][:80] if final else prompt[:80]
            server.add_history_item(title, messages)
            self.history = server.load_history()
            self.populate_history_list()

        self.worker.partial.connect(on_partial)
        self.worker.finished.connect(on_finished)
        self.worker.start()

    def on_mic(self):
        self.stt_worker = STTWorker()
        self.stt_worker.result.connect(self.on_stt_result)
        self.stt_worker.error.connect(self.on_stt_error)
        self.stt_worker.start()

    def on_stt_result(self, text):
        prev = self.input.toPlainText()
        if prev and not prev.endswith(" "):
            prev += " "
        self.input.setPlainText(prev + text)

    def on_stt_error(self, err):
        print("[stt error]", err)

    def populate_history_list(self):
        self.history_list.clear()
        self.history = server.load_history()
        for item in self.history:
            title = item.get("title", "(no title)")
            lw = QListWidgetItem(title)
            self.history_list.addItem(lw)

    def on_history_click(self, item: QListWidgetItem):
        idx = self.history_list.row(item)
        if idx < 0 or idx >= len(self.history):
            return
        entry = self.history[idx]
        messages = entry.get("messages", [])
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                self.append_user(content)
            else:
                lbl, copy_btn = self.append_assistant_streaming(content)
                copy_btn.clicked.connect(lambda text=content: QGuiApplication.clipboard().setText(text))


def main():
    app = QApplication(sys.argv)
    qss = os.path.join(os.path.dirname(__file__), "styles.qss")
    if os.path.exists(qss):
        app.setStyleSheet(open(qss, 'r', encoding='utf-8').read())
    w = ChatWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
