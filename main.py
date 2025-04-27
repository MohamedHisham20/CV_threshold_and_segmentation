from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QScrollArea, QFileDialog, QRadioButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import sys
from PyQt5 import uic
from GlobalThresholding import GlobalThresholding
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("task04.ui")  # Load the UI file
        
        ##### bc of resolution issues for my laptop, comment out if not needed!!!
        scroll = QScrollArea()
        scroll.setWidget(self.ui)
        self.setCentralWidget(scroll)
        self.resize(1360, 768)
        ######## end of resolution fix 
        
        # Find widgets
        self.load_btn = self.ui.findChild(QPushButton, "LoadImage")
        self.input_label = self.ui.findChild(QLabel, "InputImage")
        self.output_label = self.ui.findChild(QLabel, "OutputImage")
        self.otsu_check = self.ui.findChild(QRadioButton, "otsu")
        self.optimal_check = self.ui.findChild(QRadioButton, "Optimal")

        self.input_label.setScaledContents(True)
        self.output_label.setScaledContents(True)

        # Connect buttons
        self.load_btn.clicked.connect(self.load_image)
        self.otsu_check.toggled.connect(self.apply_global)
        self.optimal_check.toggled.connect(self.apply_global)

        # Variables to hold images
        self.original_image = None  # colored image
        self.gray_image = None      # grayscale image

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.original_image = cv2.imread(file_name)
            if self.original_image is None:
                print("Error loading image")
                return

            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # Show input image
            rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.input_label.setPixmap(QPixmap.fromImage(q_image))

            # Clear output image
            self.output_label.clear()

    def apply_global(self):
        if self.gray_image is None:
            return

        thresholding = GlobalThresholding(self.gray_image)

        if self.otsu_check.isChecked():
            result = thresholding.otsu_thresholding()
        elif self.optimal_check.isChecked():
            result = thresholding.optimal_thresholding(initial_threshold=128, tolerance=0.5)
        else:
            return

        # Display output image
        height, width = result.shape
        q_image = QImage(result.data, width, height, width, QImage.Format_Grayscale8)
        self.output_label.setPixmap(QPixmap.fromImage(q_image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
