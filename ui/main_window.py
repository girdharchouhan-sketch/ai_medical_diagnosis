from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QLabel,
    QHBoxLayout,
    QFrame
)

from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from ui.xray_tab import XRayTab


class MedicalDiagnosisApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "AI Medical Diagnosis Assistant"
        )

        self.setGeometry(
            100,
            100,
            1600,
            900
        )

        self.setStyleSheet("""

            QMainWindow{
                background-color: #0f172a;
            }

            QTabWidget::pane{
                border: none;
            }

            QTabBar::tab{
                background: #1e293b;
                color: white;
                padding: 15px;
                border-radius: 10px;
                min-width: 150px;
                font-size: 15px;
                font-weight: bold;
            }

            QTabBar::tab:selected{
                background: #2563eb;
            }

            QLabel{
                color: white;
            }

        """)

        # ================= HEADER =================

        header = QLabel(
            "AI Medical Diagnosis Dashboard"
        )

        header.setFont(
            QFont("Segoe UI", 28, QFont.Bold)
        )

        header.setAlignment(
            Qt.AlignCenter
        )

        header.setStyleSheet("""
            color: #38bdf8;
            padding: 20px;
        """)

        # ================= TABS =================

        self.tabs = QTabWidget()

        self.tabs.addTab(
            XRayTab(),
            "Chest X-Ray AI"
        )

        # ================= MAIN CONTAINER =================

        container = QWidget()

        main_layout = QVBoxLayout()

        main_layout.addWidget(header)

        main_layout.addWidget(self.tabs)

        container.setLayout(main_layout)

        self.setCentralWidget(container)