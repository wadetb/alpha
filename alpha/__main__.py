import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QListWidget, QListWidgetItem, QLabel, QSizePolicy
from PySide6.QtGui import QIcon, QClipboard, QGuiApplication
from PySide6.QtCore import Signal, Slot, QThread, QObject, QByteArray, QSize
import json

# Discussion of fine-tuning the whisper model:
# https://github.com/openai/whisper/discussions/759

# For the mouse:
# https://bbs.archlinux.org/viewtopic.php?id=145502
# https://openrazer.github.io/


class HistoryListItem(QListWidgetItem):
    def __init__(self, text, listWidget, parent):
        super().__init__(text, listWidget)
        self.listWidget = listWidget
        self.parent = parent

        self.widget = QWidget()

        self.layout = QHBoxLayout()
        # self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)

        self.button = QPushButton("×")
        self.button.clicked.connect(self.delete_item)

        self.label = QLabel(text)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # Make label expand

        self.layout.addWidget(self.label, stretch=1)
        self.layout.addWidget(self.button, stretch=0)

        self.widget.setLayout(self.layout)

        self.listWidget.setItemWidget(self, self.widget)

    def delete_item(self):
        item_index = self.listWidget.indexFromItem(self).row()

        self.listWidget.takeItem(item_index)
        self.parent.save_history()


class SpeechToTextApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = whisper.load_model("base")
        self.load_history()

    def initUI(self):
        # set the app icon
        self.setWindowIcon(QIcon('alpha.png'))

        self.recordedListWidget = QListWidget()
        self.recordedListWidget.setHidden(True) # not ready yet
        # self.recordedListWidget.setStyleSheet("""
        #     QListWidget::item {
        #         border: 1px solid black;
        #         margin: 2px;
        #         padding: 2px;
        #     }
        # """)

        self.textEdit = QTextEdit()

        # self.clearButton = QPushButton("Clear")
        # self.clearButton.clicked.connect(self.clear_recorded)

        self.recordButton = QPushButton("Record")
        self.recordButton.clicked.connect(self.toggle_recording)
        self.recordButton.setCheckable(True) 
        self.recordButton.setStyleSheet("""
            QPushButton:checked {
                background-color: red;
                border-style: inset;
                padding: 4px;
            }
        """)

        self.copyToClipButton = QPushButton("→ Clip")
        self.copyToClipButton.clicked.connect(self.copy_to_clipboard)
        # self.copyFromClipButton = QPushButton("← Clip")
        # self.copyFromClipButton.clicked.connect(self.copy_from_clipboard)

        self.buttonsLayout = QVBoxLayout()
        # self.buttonsLayout.addWidget(self.clearButton)
        self.buttonsLayout.addWidget(self.recordButton)
        self.buttonsLayout.addWidget(self.copyToClipButton)
        # self.buttonsLayout.addWidget(self.copyFromClipButton)
        self.buttonsLayout.addStretch()

        self.hLayout = QHBoxLayout()
        self.hLayout.addWidget(self.textEdit)
        self.hLayout.addLayout(self.buttonsLayout)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.recordedListWidget)  # Add the QListWidget to the layout
        self.layout.addLayout(self.hLayout)

        self.setLayout(self.layout)
        self.setWindowTitle('Alpha')
        
        self.recording = False
    
    def load_history(self):
        try:
            with open('history.json', 'r') as file:
                history = json.load(file)
                for text in history:
                    HistoryListItem(text, self.recordedListWidget, self)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # No history file or file is empty

    def save_history(self):
        history = [self.recordedListWidget.item(i).text() for i in range(self.recordedListWidget.count())]
        if self.textEdit.toPlainText() != "":
            history.append(self.textEdit.toPlainText())
        with open('history.json', 'w') as file:
            json.dump(history, file)

    def clear_recorded(self):
        self.textEdit.setPlainText("")

    def copy_to_clipboard(self):
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.textEdit.toPlainText())

    def copy_from_clipboard(self):
        clipboard = QGuiApplication.clipboard()
        self.textEdit.setPlainText(clipboard.text())

    def toggle_recording(self):
        if self.recording:
            self.recording = False

            if self.textEdit.toPlainText() != "":
                HistoryListItem(self.textEdit.toPlainText(), self.recordedListWidget, self)
                self.save_history()

            self.stream.close()
            self.streamFile.close()

            audio = whisper.load_audio("recording.wav")
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            options = whisper.DecodingOptions()
            result = whisper.decode(self.model, mel, options)

            cursor = self.textEdit.textCursor()
            if cursor.hasSelection():
                cursor.beginEditBlock()
                cursor.removeSelectedText()
                cursor.insertText(result.text)
                cursor.endEditBlock()
                self.textEdit.setTextCursor(cursor)
            else:
                self.textEdit.setPlainText(result.text)

            self.copy_to_clipboard()
            
        else:
            self.recording = True

            self.streamFile = sf.SoundFile("recording.wav", mode="w", samplerate=16000, channels=1)
            self.stream = sd.InputStream(callback=self.audio_callback, samplerate=16000, channels=1)
            self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)

        if self.recording:
            self.streamFile.write(indata)


def main():
    app = QApplication(sys.argv)
    ex = SpeechToTextApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
