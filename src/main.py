import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QFile
from interface_ui import Ui_mainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setFixedSize(756, 695)
        self.isRecording = False
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)


    def handle_recording(self):
        if not self.isRecording:
            print("Lancement de l'enregistrement")
            self.isRecording = True
        else:
            print("Arret de l'enregistrement")
            self.isRecording = False


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())