import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QComboBox, QTextEdit, QFileDialog, QMessageBox,
                            QGroupBox, QProgressBar, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPalette, QColor, QFont
from model import OpenAIModel, GLMModel
from translator import PDFTranslator
from utils import LOG

class TranslatorThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, pdf_path, file_format):
        super().__init__()
        self.model = model
        self.pdf_path = pdf_path
        self.file_format = file_format

    def run(self):
        try:
            translator = PDFTranslator(self.model)
            translated_text = translator.translate_pdf(
                self.pdf_path, 
                self.file_format,
                progress_callback=lambda text: self.progress.emit(text)
            )
            self.finished.emit(translated_text)
        except Exception as e:
            self.error.emit(str(e))

class StyledButton(QPushButton):
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.setMinimumHeight(35)
        self.setFont(QFont("Segoe UI", 10))
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:disabled {
                    background-color: #BDBDBD;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #E0E0E0;
                    color: #212121;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: #BDBDBD;
                }
                QPushButton:disabled {
                    background-color: #F5F5F5;
                }
            """)

class StyledGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setStyleSheet("""
            QGroupBox {
                background-color: #FFFFFF;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 1em;
                padding-top: 10px;
                font-weight: bold;
                color: #212121;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)

class TranslatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Translator")
        self.setMinimumSize(1000, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QLabel {
                color: #212121;
                font-size: 10pt;
            }
            QLineEdit, QComboBox {
                padding: 5px;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                background-color: white;
                min-height: 25px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 2px solid #2196F3;
            }
            QTextEdit {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                text-align: center;
                background-color: #F5F5F5;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
        """)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Create splitter for status and translation display
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # Top section (settings)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setSpacing(10)

        # Model settings group
        model_group = StyledGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)

        # Model type selection
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["OpenAIModel", "GLMModel"])
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_type_layout.addWidget(self.model_type_combo)
        model_layout.addLayout(model_type_layout)

        # OpenAI settings
        self.openai_group = StyledGroupBox("OpenAI Settings")
        openai_layout = QVBoxLayout()
        openai_layout.setSpacing(10)
        
        # Model name
        model_name_layout = QHBoxLayout()
        model_name_layout.addWidget(QLabel("Model Name:"))
        self.model_name_input = QLineEdit("gpt-3.5-turbo")
        model_name_layout.addWidget(self.model_name_input)
        openai_layout.addLayout(model_name_layout)

        # API Key
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_layout.addWidget(self.api_key_input)
        openai_layout.addLayout(api_key_layout)

        self.openai_group.setLayout(openai_layout)
        model_layout.addWidget(self.openai_group)

        # GLM settings
        self.glm_group = StyledGroupBox("GLM Settings")
        glm_layout = QVBoxLayout()
        glm_layout.setSpacing(10)
        
        # GLM URL
        glm_url_layout = QHBoxLayout()
        glm_url_layout.addWidget(QLabel("GLM URL:"))
        self.glm_url_input = QLineEdit()
        glm_url_layout.addWidget(self.glm_url_input)
        glm_layout.addLayout(glm_url_layout)

        self.glm_group.setLayout(glm_layout)
        self.glm_group.hide()
        model_layout.addWidget(self.glm_group)

        model_group.setLayout(model_layout)
        top_layout.addWidget(model_group)

        # File settings group
        file_group = StyledGroupBox("File Settings")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(10)

        # PDF file selection
        pdf_layout = QHBoxLayout()
        pdf_layout.addWidget(QLabel("PDF File:"))
        self.pdf_path_input = QLineEdit()
        pdf_layout.addWidget(self.pdf_path_input)
        self.pdf_browse_btn = StyledButton("Browse")
        self.pdf_browse_btn.clicked.connect(self.browse_pdf)
        pdf_layout.addWidget(self.pdf_browse_btn)
        file_layout.addLayout(pdf_layout)

        # Output format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["pdf", "markdown"])
        format_layout.addWidget(self.format_combo)
        file_layout.addLayout(format_layout)

        file_group.setLayout(file_layout)
        top_layout.addWidget(file_group)

        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()  # Add stretch to push button to center
        self.translate_btn = StyledButton("Translate", primary=True)
        self.translate_btn.setFixedWidth(200)  # Set fixed width for the button
        self.translate_btn.clicked.connect(self.start_translation)
        button_layout.addWidget(self.translate_btn)
        button_layout.addStretch()  # Add stretch to push button to center
        top_layout.addLayout(button_layout)

        # Status display
        status_group = StyledGroupBox("Translation Status")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(5)
        
        # Progress bar inside status group
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(5)  # Make it thinner
        self.progress_bar.setMaximumHeight(5)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #F5F5F5;
                border-radius: 2px;
                margin-top: 0px;
                margin-bottom: 0px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 2px;
            }
        """)
        status_layout.addWidget(self.progress_bar)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        status_layout.addWidget(self.status_text)

        status_group.setLayout(status_layout)
        top_layout.addWidget(status_group)

        # Add top section to splitter
        splitter.addWidget(top_widget)

        # Translation display
        translation_group = StyledGroupBox("Translated Text")
        translation_layout = QVBoxLayout()
        
        self.translation_text = QTextEdit()
        self.translation_text.setReadOnly(True)
        translation_layout.addWidget(self.translation_text)

        translation_group.setLayout(translation_layout)
        splitter.addWidget(translation_group)

        # Set initial splitter sizes
        splitter.setSizes([300, 500])

        # Initialize translator thread
        self.translator_thread = None

    def on_model_type_changed(self, model_type):
        if model_type == "OpenAIModel":
            self.openai_group.show()
            self.glm_group.hide()
        else:
            self.openai_group.hide()
            self.glm_group.show()

    def browse_pdf(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF File",
            "",
            "PDF Files (*.pdf)"
        )
        if file_name:
            self.pdf_path_input.setText(file_name)

    def start_translation(self):
        # Clear previous status and translation
        self.status_text.clear()
        self.translation_text.clear()
        
        # Validate inputs
        if not self.pdf_path_input.text():
            QMessageBox.warning(self, "Error", "Please select a PDF file")
            return

        if self.model_type_combo.currentText() == "OpenAIModel":
            if not self.api_key_input.text():
                QMessageBox.warning(self, "Error", "Please enter OpenAI API key")
                return
            model = OpenAIModel(
                model=self.model_name_input.text(),
                api_key=self.api_key_input.text()
            )
        else:
            if not self.glm_url_input.text():
                QMessageBox.warning(self, "Error", "Please enter GLM URL")
                return
            model = GLMModel()

        # Disable UI elements
        self.translate_btn.setEnabled(False)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setValue(0)

        # Start translation thread
        self.translator_thread = TranslatorThread(
            model,
            self.pdf_path_input.text(),
            self.format_combo.currentText()
        )
        self.translator_thread.progress.connect(self.update_status)
        self.translator_thread.finished.connect(self.translation_finished)
        self.translator_thread.error.connect(self.translation_error)
        self.translator_thread.start()

    def update_status(self, text):
        self.status_text.append(text)
        # Auto-scroll to the bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )

    def translation_finished(self, translated_text):
        self.translate_btn.setEnabled(True)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        self.translation_text.setText(translated_text)
        QMessageBox.information(self, "Success", "Translation completed!")

    def translation_error(self, error_message):
        self.translate_btn.setEnabled(True)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Error", f"Translation failed: {error_message}")

def main():
    app = QApplication(sys.argv)
    window = TranslatorGUI()
    window.show()
    sys.exit(app.exec()) 