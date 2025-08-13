


from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QPushButton, QLineEdit, QLabel, QFileDialog
)   

def apply_dark_theme(self):
    # """Apply a dark theme using Qt Style Sheets."""
    # dark_stylesheet = """
    # QWidget {
    #     background-color: #2b2b2b;
    #     color: #ffffff;
    # }
    # QPushButton {
    #     background-color: #444444;
    #     color: #ffffff;
    #     border: 1px solid #777777;
    #     padding: 5px;
    # }
    # QPushButton:checked {
    #     background-color: #6A5ACD;
    #     border: 1px solid #aaaaaa;
    # }
    # QLineEdit {
    #     background-color: #3a3a3a;
    #     color: #ffffff;
    #     border: 1px solid #777777;
    #     padding: 3px;
    # }
    # QLabel {
    #     font-weight: bold;
    #     color: #ffffff;
    # }
    # """
    dark_stylesheet = """
    QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    QPushButton {
        background-color: #444444;
        color: #ffffff;
        border: 1px solid #777777;
        padding: 5px;
    }
    QPushButton:hover {
        background-color: #555555;
    }
    QPushButton:pressed {
        background-color: #333333;
    }
    QPushButton:checked {
        background-color: #6A5ACD;
        border: 1px solid #aaaaaa;
    }
    QLineEdit {
        background-color: #3a3a3a;
        color: #ffffff;
        border: 1px solid #777777;
        padding: 3px;
    }
    QLabel {
        font-weight: bold;
        color: #ffffff;
    }
    """

    self.setStyleSheet(dark_stylesheet)