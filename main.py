import sys

from PyQt5.QtWidgets import QApplication

from ui.main_window import MedicalDiagnosisApp


if __name__ == "__main__":

    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = MedicalDiagnosisApp()

    window.showMaximized()

    sys.exit(app.exec_())