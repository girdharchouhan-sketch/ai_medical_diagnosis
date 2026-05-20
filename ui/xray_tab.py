from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QFrame
)

from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

from models.inference import predict_xray
from utils.gradcam import generate_gradcam


class XRayTab(QWidget):

    def __init__(self):
        super().__init__()

        self.setStyleSheet("""

            QWidget{
                background-color: #0f172a;
                color: white;
                font-family: Segoe UI;
            }

            QFrame{
                background-color: #1e293b;
                border-radius: 20px;
            }

            QPushButton{
                background-color: #2563eb;
                color: white;
                border-radius: 12px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
            }

            QPushButton:hover{
                background-color: #1d4ed8;
            }

            QLabel{
                color: white;
                font-size: 16px;
            }

        """)

        # ================= LEFT PANEL =================

        self.left_frame = QFrame()

        left_layout = QVBoxLayout()

        self.image_label = QLabel("Upload Chest X-Ray")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.image_label.setFixedSize(500, 500)

        self.image_label.setStyleSheet("""
            QLabel{
                border: 3px dashed #334155;
                border-radius: 20px;
                background-color: #111827;
            }
        """)

        self.upload_button = QPushButton(
            "Upload X-Ray"
        )

        self.upload_button.clicked.connect(
            self.select_image
        )

        left_layout.addWidget(
            self.image_label
        )

        left_layout.addSpacing(20)

        left_layout.addWidget(
            self.upload_button
        )

        self.left_frame.setLayout(
            left_layout
        )

        # ================= RIGHT PANEL =================

        self.right_frame = QFrame()

        right_layout = QVBoxLayout()

        title = QLabel(
            "AI Diagnosis Result"
        )

        title.setFont(
            QFont("Segoe UI", 24, QFont.Bold)
        )

        title.setStyleSheet("""
            color: #38bdf8;
        """)

        self.result_label = QLabel(
            "Diagnosis: Waiting..."
        )

        self.result_label.setFont(
            QFont("Segoe UI", 20)
        )

        self.result_label.setStyleSheet("""
            color: #f87171;
        """)

        self.confidence_label = QLabel(
            "Confidence: 0%"
        )

        self.confidence_label.setFont(
            QFont("Segoe UI", 18)
        )

        self.status_label = QLabel(
            "System Status: Ready"
        )

        self.status_label.setFont(
            QFont("Segoe UI", 16)
        )

        self.status_label.setStyleSheet("""
            color: #4ade80;
        """)

        right_layout.addSpacing(50)

        right_layout.addWidget(title)

        right_layout.addSpacing(80)

        right_layout.addWidget(
            self.result_label
        )

        right_layout.addSpacing(50)

        right_layout.addWidget(
            self.confidence_label
        )

        right_layout.addSpacing(50)

        right_layout.addWidget(
            self.status_label
        )

        right_layout.addStretch()

        self.right_frame.setLayout(
            right_layout
        )

        # ================= MAIN LAYOUT =================

        main_layout = QHBoxLayout()

        main_layout.addWidget(
            self.left_frame,
            2
        )

        main_layout.addWidget(
            self.right_frame,
            1
        )

        self.setLayout(main_layout)

    def select_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select X-Ray Image",
            "",
            "Image Files (*.png *.jpg *.jpeg)"
        )

        if not file_path:
            return

        try:

            self.status_label.setText(
                "System Status: Processing..."
            )

            prediction, confidence = predict_xray(
                file_path
            )

            heatmap = generate_gradcam(
                file_path
            )

            pixmap = QPixmap(heatmap)

            self.image_label.setPixmap(
                pixmap.scaled(
                    500,
                    500,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

            self.result_label.setText(
                f"Diagnosis: {prediction}"
            )

            self.confidence_label.setText(
                f"Confidence: {confidence:.2f}%"
            )

            self.status_label.setText(
                "System Status: Completed"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Prediction Error",
                str(e)
            )