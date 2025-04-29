## main.py
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QScrollArea, \
    QFileDialog, QRadioButton, QButtonGroup, QMessageBox, QSpinBox, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import sys
from PyQt5 import uic

from AgglomerativeClustering import agglomerate_clusters, cv2_to_qimage_agglomerate
from OtsuAndOptimal import OtsuAndOptimal  
import numpy as np

from RegionGrowingSegmentation import RegionGrowingDialog, cv2_to_qimage
from KMeanSegmentation import kmeans, kmeans_result_to_qimage
from MeanShiftClustering import MeanShiftClusterer


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
        self.kmeans_check = self.ui.findChild(QRadioButton, "kMeans")

        self.agglomerative_check = self.ui.findChild(QRadioButton, "agglomerative")
        self.meanShift_check = self.ui.findChild(QRadioButton, "meanShift")

        self.iterations_slider = self.ui.findChild(QSlider, "iterationsHorizontalSlider")
        self.iterationsLabel = self.ui.findChild(QLabel, "iterationsLabel")
        
        self.mode_gp = QButtonGroup(self)
        self.mode_gp.setExclusive(True)
        self.mode_gp.addButton(self.local_check)
        self.mode_gp.addButton(self.global_check)

        self.window_size = 13
       
        self.input_label.setScaledContents(True)
        self.output_label.setScaledContents(True)

        # Connect buttons
        self.load_btn.clicked.connect(self.load_image)
        self.otsu_check.toggled.connect(self.check_thresholding_mode)
        self.optimal_check.toggled.connect(self.check_thresholding_mode)

        # Connect radio buttons
        self.regionGrowing_check.toggled.connect(self.apply_region_growing)
        self.kmeans_check.toggled.connect(self.apply_kmeans)

        self.agglomerative_check.toggled.connect(self.apply_agglomerative)
        self.meanShift_check.toggled.connect(self.apply_meanShift)

        self.iterations_slider.setMinimum(1)
        self.iterations_slider.setMaximum(100)
        self.iterations_slider.setValue(10)
        self.iterations_slider.valueChanged.connect(self.update_iterations_label)

        # Variables to hold images
        self.original_image = None  # colored Q image
        self.gray_image = None      # grayscale image
        self.rgb_image = None       # colored np image for displaying

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
            self.rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            height, width, channel = self.rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
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
            self.window_size = self.windowsize_spinbox.value()
            result = thresholding.local_otsu(self.window_size)
            message = "Applied Local Otsu Thresholding"
        elif self.optimal_check.isChecked():
            self.window_size = self.windowsize_spinbox.value()
            result = thresholding.local_optimal(self.window_size)
            message = "Applied Local Optimal Thresholding"
        else:
            return

        # result = cv2.GaussianBlur(result, (5, 5), 0)
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
        if self.regionGrowing_check.isChecked():
            self.kmeans_check.setChecked(False)

            if self.original_image is None:
                self.show_message("Please load an image first")
                return

            # Create the dialog with the current image
            dialog = RegionGrowingDialog(self, self.original_image)


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

    def apply_kmeans(self):
        if self.kmeans_check.isChecked():
            self.regionGrowing_check.setChecked(False)
            # Check if an image is loaded
            if self.original_image is None:
                self.show_message("Please load an image first")
                return

            result_image = kmeans(self.original_image,k=3, max_iters=self.iterations_slider.value())
            # Convert the result image to QImage
            q_image = kmeans_result_to_qimage(result_image)
            # Display the result image
            self.output_label.setPixmap(QPixmap.fromImage(q_image))
            self.show_message("K-means segmentation completed!")

    def apply_meanShift(self):
        if self.meanShift_check.isChecked():
            if self.original_image is None:
                self.show_message("Please load an image first")
                return

            bandwidth = 100
            with_spatial_coords = False
            clusterer = MeanShiftClusterer(self.original_image, bandwidth=bandwidth, with_spatial_coords=with_spatial_coords)
            clusterer.cluster()

            clustered_image = clusterer.get_clustered_image()
            q_image = cv2_to_qimage(clustered_image)

            if q_image.isNull():
                self.show_message("Error: Generated QImage is null!")
                return

            # Display the result image
            self.output_label.setPixmap(QPixmap.fromImage(q_image))
            self.show_message("Mean Shift clustering completed!")

    def update_iterations_label(self):
        # Get the current value of the slider
        iterations = self.iterations_slider.value()
        # Update the label or any other UI element to show the current value
        self.ui.iterationsLabel.setText(f"{iterations}")

    def apply_agglomerative(self):
        if self.agglomerative_check.isChecked():
            self.regionGrowing_check.setChecked(False)
            self.kmeans_check.setChecked(False)

            # Check if an image is loaded
            if self.original_image is None:
                self.show_message("Please load an image first")
                return

            try:
                # Get image in correct format
                image_to_process = self.original_image.copy()
                if len(image_to_process.shape) == 2:  # If grayscale, convert to RGB
                    image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_GRAY2RGB)
                elif image_to_process.shape[2] == 3:  # If BGR, convert to RGB
                    image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)

                # Apply agglomerative clustering with progress updates
                self.show_message("Starting agglomerative clustering. This may take a while...")
                num_clusters = 5  # You might want to make this configurable with a slider

                # Apply clustering with optimized parameters
                clustered_image = agglomerate_clusters(
                    image_to_process,
                    num_clusters=num_clusters,
                    color_weight=2.0,  # Give more weight to color differences
                    spatial_weight=1.0
                )

                # Convert the output back to BGR for cv2 display if needed
                output_bgr = cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR)

                # Convert to QImage
                height, width, channel = output_bgr.shape
                bytesPerLine = 3 * width
                qImg = QImage(output_bgr.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

                # Display the result
                self.output_label.setPixmap(QPixmap.fromImage(qImg))
                self.show_message("Agglomerative clustering completed!")

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.show_message(f"Error in agglomerative clustering: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
