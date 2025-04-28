## main.py
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QScrollArea, QFileDialog, QRadioButton, QButtonGroup, QMessageBox, QSpinBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import sys
from PyQt5 import uic
from OtsuAndOptimal import OtsuAndOptimal  
import numpy as np

from seed_widget import RegionGrowingDialog


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
        self.local_check = self.ui.findChild(QRadioButton, "local")
        self.global_check = self.ui.findChild(QRadioButton, "global")
        self.windowsize_spinbox = self.ui.findChild(QSpinBox, "windowsize")

        self.regionGrowing_check = self.ui.findChild(QRadioButton, "regionGrowing")
        
        self.mode_gp = QButtonGroup(self)
        self.mode_gp.setExclusive(True)
        self.mode_gp.addButton(self.local_check)
        self.mode_gp.addButton(self.global_check)

        self.window_size = self.windowsize_spinbox.value()
       
        self.input_label.setScaledContents(True)
        self.output_label.setScaledContents(True)

        # Connect buttons
        self.load_btn.clicked.connect(self.load_image)
        self.otsu_check.toggled.connect(self.check_thresholding_mode)
        self.optimal_check.toggled.connect(self.check_thresholding_mode)

        # Connect radio buttons
        self.regionGrowing_check.toggled.connect(self.apply_region_growing)

        # Variables to hold images
        self.original_image = None  # colored image
        self.gray_image = None      # grayscale image

    def show_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Thresholding Status")
        msg.exec_()

    
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        self.path = file_name
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
            
    def check_thresholding_mode(self):
        if self.local_check.isChecked():
            self.global_check.setChecked(False)
            return self.apply_local()
        elif self.global_check.isChecked():
            self.local_check.setChecked(False)
            return self.apply_global()
            
    def apply_local(self):
        if self.gray_image is None:
            return

        thresholding = OtsuAndOptimal(self.path)

        if self.otsu_check.isChecked():
            result = thresholding.local_otsu(self.window_size)
            message = "Applied Local Otsu Thresholding"
        elif self.optimal_check.isChecked():
            result = thresholding.local_optimal(self.window_size)
            message = "Applied Local Optimal Thresholding"
        else:
            return

        result = cv2.GaussianBlur(result, (5, 5), 0)
        # Display output image
        height, width = result.shape
        q_image = QImage(result.data, width, height, width, QImage.Format_Grayscale8)
        self.output_label.setPixmap(QPixmap.fromImage(q_image))
        self.show_message(message)


    def apply_global(self):
        if self.gray_image is None:
            return

        thresholding = OtsuAndOptimal(self.path)

        if self.otsu_check.isChecked():
            result = thresholding.otsu_thresholding()
            message = "Applied Global Otsu Thresholding"
        elif self.optimal_check.isChecked():
            result = thresholding.optimal_thresholding(initial_threshold=128, tolerance=0.5)
            message = "Applied Global Optimal Thresholding"
        else:
            return

        # Display output image
        height, width = result.shape
        q_image = QImage(result.data, width, height, width, QImage.Format_Grayscale8)
        self.output_label.setPixmap(QPixmap.fromImage(q_image))
        self.show_message(message)

    def apply_region_growing(self):
        if self.original_image is None:
            self.show_message("Please load an image first")
            return

        # # Create the dialog and pass the current image
        # rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        # height, width, channel = rgb_image.shape
        # bytes_per_line = 3 * width
        # q_image = QImage(rgb_image.data.copy(), width, height, bytes_per_line, QImage.Format_RGB888)

        # print('before dialoge')
        # Create the dialog with the current image
        dialog = RegionGrowingDialog(self, self.original_image)
        # print('after dialoge')

        # Connect signal to handle the result
        dialog.segmentationCompleted.connect(self.handle_region_growing_result)

        # Show dialog as modal (blocks interaction with main window until closed)
        dialog.exec_()

    def handle_region_growing_result(self, result_image):
        # Convert QImage to pixmap and set it in the output label
        # plot the result image

        pixmap = QPixmap.fromImage(result_image)

        self.output_label.setPixmap(pixmap)
        self.show_message("Region growing segmentation completed!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
