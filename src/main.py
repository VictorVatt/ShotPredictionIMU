import sys
import time

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QFile, QTimer, QTime
from interface_ui import Ui_mainWindow
import numpy as np
from mat4py import loadmat
import matplotlib.pyplot as plt


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setFixedSize(756, 695)
        self.isRecording = False
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showTime)
        self.ui.recordButton.clicked.connect(self.startTimer)
    def showTime(self):
        t = QTime(0, 0, 0).addSecs(1)
        text = t.toString("hh:mm:ss")
        self.ui.timer_ldc.display(text)

    def startTimer(self):
        self.timer.start(1)

    def endTimer(self):
        self.timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())