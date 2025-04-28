import cv2
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage
import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QFileDialog, QSlider)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


# select k initial centroids for rgb image segmentation
def create_random_centroids(pixels, k):
    # m is number of pixels (rows), and n would be 3 (r,g,b)
    m, n = pixels.shape
    # Initialize k centroids random from the pixels with 3 channels
    centroids = np.zeros((k, n))
    for i in range(k):
        # Randomly select a pixel from the image
        rand_index = np.random.randint(0, m)
        # Assign the pixel value to the centroid
        centroids[i] = pixels[rand_index]
    return centroids


def kmeans(image, k, max_iters=100):
    if len(image.shape) == 3:
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3)
    else:
        # If the image is grayscale, we need to reshape it to 2D
        pixels = image.reshape(-1, 1)

    # m is number of pixels (rows), and n would be 3 (r,g,b)
    m, n = pixels.shape

    # Initialize centroids
    centroids = create_random_centroids(pixels, k)

    # Initialize variables
    prev_centroids = np.zeros(centroids.shape)
    idx = np.zeros(m, dtype=int)  # integer indexing

    for _ in range(max_iters):
        print(f"Iteration {_ + 1}/{max_iters}")
        # Assign clusters
        for i in range(m):
            distances = np.linalg.norm(pixels[i] - centroids, axis=1)
            # Find the index of the closest centroid
            idx[i] = np.argmin(distances)

        # Update centroids
        prev_centroids = centroids.copy()
        for i in range(k):
            # Get all points assigned to the i-th centroid
            points_in_cluster = pixels[idx == i]
            # If there are points in the cluster, update the centroid
            if len(points_in_cluster) > 0:
                centroids[i] = np.mean(points_in_cluster, axis=0)

        # Check for convergence
        # because they might be close enough but not exactly equal
        if np.allclose(centroids, prev_centroids):
            print(f"Converged at {_} iterations")
            break

    # Build segmented image
    # Assign each pixel to the nearest centroid
    segmented_image = centroids[idx.astype(int)]
    # Reshape back to original image shape
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


def qimage_to_cv2(qimage):
    qformat = qimage.format()

    if qformat == QImage.Format_Grayscale8:
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width))
        return arr
    else:
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr


def rgb_to_luv_vectorized(image):
    # Normalize RGB to [0,1]
    rgb = image.astype(float) / 255.0

    # Apply gamma correction
    mask = rgb <= 0.04045
    rgb[mask] /= 12.92
    rgb[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4

    # RGB to XYZ matrix
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    X = r * 0.412453 + g * 0.357580 + b * 0.180423
    Y = r * 0.212671 + g * 0.715160 + b * 0.072169
    Z = r * 0.019334 + g * 0.119193 + b * 0.950227

    # Constants for white point D65
    Yn = 1.0
    xn, yn = 0.312713, 0.329016
    un = 4 * xn / (xn + 15 * yn + 3 * (1 - xn - yn))
    vn = 9 * yn / (xn + 15 * yn + 3 * (1 - xn - yn))

    # Avoid division by zero
    epsilon = 1e-10
    denominator = X + 15 * Y + 3 * Z
    denominator = np.maximum(denominator, epsilon)

    # Calculate u' and v'
    u = 4 * X / denominator
    v = 9 * Y / denominator

    # Calculate L*
    Y_ratio = Y / Yn
    mask = Y_ratio > 0.008856
    L = np.zeros_like(Y_ratio)
    L[mask] = 116 * (Y_ratio[mask] ** (1 / 3)) - 16
    L[~mask] = 903.3 * Y_ratio[~mask]

    # Calculate u* and v*
    U = 13 * L * (u - un)
    V = 13 * L * (v - vn)

    # Stack channels
    luv_image = np.stack([L, U, V], axis=2)
    return luv_image


def region_growing(image, seeds, threshold=50, use_luv=True):
    """
    Perform region growing segmentation with optional CIELUV color space support

    Args:
        image: NumPy array of image data (RGB or grayscale)
        seeds: List of (x, y) tuples as starting points
        threshold: Distance threshold for region inclusion
        use_luv: Boolean to use CIELUV color space (default: True)

    Returns:
        NumPy array of binary segmentation mask
    """
    # convert qimage to cv2 image
    if isinstance(image, QImage):
        image = qimage_to_cv2(image)

    # Handle color space conversion
    if len(image.shape) == 3 and use_luv:
        # Convert BGR to CIELUV for better perceptual color distance
        # Note: OpenCV expects BGR order by default
        luv_image = rgb_to_luv_vectorized(image)
        processed_image = luv_image
    elif len(image.shape) == 3 and not use_luv:
        # Keep original RGB image
        processed_image = image.copy()
    else:
        # For grayscale images
        processed_image = image.copy()
        use_luv = False  # Force use_luv to False for grayscale

    # Get dimensions based on original image
    if len(image.shape) == 3:
        height, width, _ = processed_image.shape
    else:
        height, width = processed_image.shape

    # Create segmentation mask for all seeds
    overall_segmentation = np.zeros((height, width), dtype=np.uint8)

    # Process each seed point separately
    for seed_idx, (seed_x, seed_y) in enumerate(seeds, 1):
        # Skip if out of bounds
        if 0 <= seed_x < width and 0 <= seed_y < height:
            visited = np.zeros((height, width), dtype=bool)
            seed_segment = np.zeros((height, width), dtype=np.uint8)

            # Get seed pixel value
            if len(processed_image.shape) == 3:
                seed_value = processed_image[seed_y, seed_x, :].astype(np.float32)
            else:
                seed_value = processed_image[seed_y, seed_x]

            # Initialize queue with seed
            processing_queue = [(seed_x, seed_y)]
            visited[seed_y, seed_x] = True
            seed_segment[seed_y, seed_x] = 1

            # Define 8-connected neighbors
            neighbors = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1)
            ]

            # Process queue
            while processing_queue:
                x, y = processing_queue.pop(0)

                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy

                    # Check bounds
                    if 0 <= nx < width and 0 <= ny < height:
                        # Skip if already visited
                        if not visited[ny, nx]:
                            visited[ny, nx] = True

                            # Calculate color difference based on color space
                            if len(processed_image.shape) == 3:
                                pixel_value = processed_image[ny, nx, :].astype(np.float32)
                                if use_luv:
                                    # For CIELUV, use Euclidean distance in LUV space
                                    color_diff = np.sqrt(np.sum((pixel_value - seed_value) ** 2))
                                else:
                                    # For RGB, use Euclidean distance
                                    color_diff = np.sqrt(np.sum((pixel_value - seed_value) ** 2))
                            else:
                                # For grayscale
                                pixel_value = processed_image[ny, nx]
                                color_diff = abs(float(pixel_value) - float(seed_value))

                            # Check if pixel is within threshold
                            if color_diff <= threshold:
                                seed_segment[ny, nx] = 1
                                processing_queue.append((nx, ny))

        # Add this segment to the final segmentation
        overall_segmentation = np.maximum(overall_segmentation, seed_segment * seed_idx)

    return overall_segmentation


class SeedPixelWidget(QWidget):
    # Signal to notify when seeds have changed - change from Signal to pyqtSignal
    seedsChanged = pyqtSignal(list)
    imageLoaded = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 500)
        self.setMouseTracking(True)

        # Store image and transformation properties
        self.image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0

        # Store seeds as (x, y, cluster_id) tuples
        self.seeds = []
        self.active_cluster = 1
        self.num_clusters = 3  # Default K value
        self.colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
                       QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)]

        # State variables
        self.hovering_point = None
        self.dragging_seed = None

        self.threshold = 50  # Default threshold for region growing

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Top controls
        controls = QHBoxLayout()

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        # Clear buttons
        self.clear_button = QPushButton("Clear Seeds")
        self.clear_button.clicked.connect(self.clear_seeds)

        # Run segmentation button
        self.run_button = QPushButton("Run Segmentation")
        # self.run_button.clicked.connect(self.run_segmentation)

        # Threshold slider (for region growing)
        self.threshold_label = QLabel("Region Growing Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(50)  # Default threshold value
        self.threshold_slider.setSingleStep(5)
        self.threshold_slider.valueChanged.connect(lambda value: setattr(self, 'threshold', value))

        # Add widgets to controls
        controls.addWidget(self.clear_button)
        controls.addWidget(self.run_button)
        controls.addWidget(self.threshold_label)
        controls.addWidget(self.threshold_slider)
        controls.addWidget(self.load_button)

        # Status bar for coordinates
        self.status_label = QLabel("Position: ")

        # Add layouts to main layout
        layout.addLayout(controls)
        layout.addStretch()
        layout.addWidget(self.status_label)

    def load_image(self):
        # """Load an image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            # Load the image
            self.image = QImage(file_path)
            if not self.image.isNull():
                # Set image to seed widget
                self.set_image(file_path)

    def set_image(self, image):
        """Set the image for segmentation (NumPy array or file path)"""
        if isinstance(image, str):
            # Load image from file path
            self.image = QImage(image)
            if self.image.isNull():
                raise ValueError(f"Failed to load image from {image}")
        elif isinstance(image, np.ndarray):
            # Convert numpy array to QImage
            height, width = image.shape[:2]
            if len(image.shape) == 2:  # Grayscale
                qimg = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
            else:  # RGB/RGBA
                bytes_per_line = 3 * width
                qimg = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image = qimg
        else:
            raise TypeError("Image must be a file path or NumPy array")

        # Create a display copy
        self.display_image = QPixmap.fromImage(self.image)
        self.update()

        # Emit signal that image was loaded
        self.imageLoaded.emit(self.image)

    def clear_seeds(self):
        """Remove all seed points"""
        self.seeds = []
        self.seedsChanged.emit(self.seeds)
        self.update()

    def paintEvent(self, event):
        if not self.display_image:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the image
        scaled_width = int(self.display_image.width() * self.scale_factor)
        scaled_height = int(self.display_image.height() * self.scale_factor)

        # Center the image in the widget
        x = (self.width() - scaled_width) // 2
        y = (self.height() - scaled_height) // 2

        painter.drawPixmap(x, y, scaled_width, scaled_height, self.display_image)

        # Draw seed points
        self.draw_seeds(painter, x, y)

        # Draw hover indication
        if self.hovering_point:
            painter.setPen(QPen(Qt.white, 2, Qt.DashLine))
            painter.drawEllipse(self.hovering_point, 10, 10)

    def draw_seeds(self, painter, offset_x, offset_y):
        """Draw all seed points with their cluster colors"""
        for seed_x, seed_y, cluster_id in self.seeds:
            # Convert image coordinates to widget coordinates
            x = offset_x + seed_x * self.scale_factor
            y = offset_y + seed_y * self.scale_factor

            # Select color based on cluster ID (modulo number of colors)
            color_index = (cluster_id - 1) % len(self.colors)
            color = self.colors[color_index]

            # Draw seed marker
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), 8, 8)

    def mousePressEvent(self, event):
        if not self.display_image:
            return

        # Check if clicking on an existing seed
        for i, (seed_x, seed_y, cluster_id) in enumerate(self.seeds):
            # Convert to widget coordinates
            x = (self.width() - self.display_image.width() * self.scale_factor) // 2 + seed_x * self.scale_factor
            y = (self.height() - self.display_image.height() * self.scale_factor) // 2 + seed_y * self.scale_factor

            # Check if click is within seed marker
            # PyQt5 uses pos() instead of position()
            if (QPointF(x, y) - event.pos()).manhattanLength() < 10:
                if event.button() == Qt.LeftButton:
                    self.dragging_seed = i
                elif event.button() == Qt.RightButton:
                    # Remove seed on right click
                    self.seeds.pop(i)
                    self.seedsChanged.emit(self.seeds)
                    self.update()
                return

        # If not clicking on existing seed, add new seed
        if event.button() == Qt.LeftButton:
            # Convert widget coordinates to image coordinates
            offset_x = (self.width() - self.display_image.width() * self.scale_factor) // 2
            offset_y = (self.height() - self.display_image.height() * self.scale_factor) // 2

            # PyQt5 uses pos() instead of position()
            image_x = (event.pos().x() - offset_x) / self.scale_factor
            image_y = (event.pos().y() - offset_y) / self.scale_factor

            # Check if coordinates are within image bounds
            if (0 <= image_x < self.display_image.width() and
                    0 <= image_y < self.display_image.height()):
                # For region growing, use cluster_id=1 for all seeds
                cluster_id = 1

                self.seeds.append((int(image_x), int(image_y), cluster_id))
                self.seedsChanged.emit(self.seeds)
                self.update()

    def mouseMoveEvent(self, event):
        if not self.display_image:
            return

        # Update status bar with position
        offset_x = (self.width() - self.display_image.width() * self.scale_factor) // 2
        offset_y = (self.height() - self.display_image.height() * self.scale_factor) // 2

        # PyQt5 uses pos() instead of position()
        image_x = (event.pos().x() - offset_x) / self.scale_factor
        image_y = (event.pos().y() - offset_y) / self.scale_factor

        if (0 <= image_x < self.display_image.width() and
                0 <= image_y < self.display_image.height()):
            # Get pixel value at current position (if needed)
            self.status_label.setText(f"Position: ({int(image_x)}, {int(image_y)})")
            self.hovering_point = event.pos()
        else:
            self.status_label.setText("Position: Outside image")
            self.hovering_point = None

        # Update dragging seed if any
        if self.dragging_seed is not None:
            if (0 <= image_x < self.display_image.width() and
                    0 <= image_y < self.display_image.height()):
                seed = self.seeds[self.dragging_seed]
                self.seeds[self.dragging_seed] = (int(image_x), int(image_y), seed[2])
                self.seedsChanged.emit(self.seeds)

        self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_seed = None

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.1

        # Zoom in/out based on wheel direction
        # PyQt5's wheel event behaves differently
        if event.angleDelta().y() > 0:
            self.scale_factor = min(self.scale_factor * zoom_factor, self.max_scale)
        else:
            self.scale_factor = max(self.scale_factor / zoom_factor, self.min_scale)

        self.update()

    def get_seed_pixels(self):
        """Return seed pixels in a format suitable for algorithms"""
        # For region growing: just list of (x, y) points
        return [(x, y) for x, y, _ in self.seeds]

    def run_segmentation(self):
        """Run the region growing algorithm and return the result"""
        if not self.seeds:
            print("no seed in run seg in seed pixel")
            return None

        print("run seg in seed pixel")

        seed_pixels = self.get_seed_pixels()
        result = region_growing(self.image, seed_pixels, threshold=self.threshold, use_luv=False)

        # Convert numpy array result to QImage for display
        height, width = result.shape[:2]
        if len(result.shape) == 2:  # Grayscale result
            bytes_per_line = width
            q_image = QImage(result.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # RGB result
            bytes_per_line = 3 * width
            q_image = QImage(result.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Update the display image
        self.display_image = QPixmap.fromImage(q_image)
        self.update()

        return q_image  # Return the QImage result

    # def run_region_growing(self):
    #     """Run Region Growing with the selected seeds"""
    #     print(f"Running Region Growing with {len(self.seeds)} seeds")
    #     seed_pixels = self.get_seed_pixels()  # Get the seed pixels as (x,y) tuples
    #     result = region_growing(self.image, seed_pixels, threshold=self.threshold, use_luv=False)
    #
    #     # Convert numpy array result to QImage for display
    #     height, width = result.shape[:2]
    #     if len(result.shape) == 2:  # Grayscale result
    #         bytes_per_line = width
    #         q_image = QImage(result.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    #     else:  # RGB result
    #         bytes_per_line = 3 * width
    #         q_image = QImage(result.data, width, height, bytes_per_line, QImage.Format_RGB888)
    #
    #     self.display_image = QPixmap.fromImage(q_image)
    #     self.update()


class RegionGrowingApp(QMainWindow):
    # Change Signal to pyqtSignal
    regionGrowingDone = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.region_growing_result = None
        self.setWindowTitle("Image Segmentation Tool")
        self.resize(1000, 700)

        self.image = None
        self.segmentation_result = None

        self.setup_ui()

    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        # Left panel - seed pixel widget
        self.seed_widget = SeedPixelWidget()
        self.seed_widget.seedsChanged.connect(self.handle_seeds_changed)
        self.seed_widget.run_button.clicked.connect(self.run_segmentation)
        self.seed_widget.imageLoaded.connect(self.update_image)  # Connect to new signal

        # Add panel to main layout
        main_layout.addWidget(self.seed_widget, 3)

    def update_image(self, image):
        """Update the main image reference when loaded in widget"""
        self.image = image

    def handle_seeds_changed(self, seeds):
        """Handle changes in seed points"""
        if self.segmentation_result:
            # Clear result when seeds change
            self.segmentation_result = None

    def run_segmentation(self):
        """Coordinate the segmentation process and handle results"""
        if not self.image or self.image.isNull():
            print("No image loaded for run seg in region grow")
            return

        print("rung seg in region grow")

        # Call the widget's segmentation function to run the algorithm
        result_image = self.seed_widget.run_segmentation()

        if result_image:
            # Store and emit the result
            self.segmentation_result = result_image
            self.regionGrowingDone.emit(result_image)
            self.close()  # Optional: close the popup after emitting

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RegionGrowingApp()
    window.show()
    sys.exit(app.exec_())

#     def display_segmentation_result(self):
#         """Display the segmentation result on top of the input image"""
#         if self.segmentation_result is None or self.image is None:
#             return
#
#         # Get dimensions from segmentation result
#         height, width = self.segmentation_result.shape
#
#         # Convert QImage image to numpy array
#         image_array = self.qimage_to_numpy(self.image)
#
#         # Resize if necessary to match segmentation dimensions
#         if image_array.shape[:2] != (height, width):
#             image_array = cv2.resize(image_array, (width, height))
#
#         # Create result image using input image as background
#         result = image_array.copy()
#
#         # Color each segment's boundary based on its label
#         unique_labels = np.unique(self.segmentation_result)
#
#         for label in unique_labels:
#             if label == 0:  # Skip background
#                 continue
#
#             # Get color for this label
#             color_idx = (label - 1) % len(self.seed_widget.colors)
#             color = self.seed_widget.colors[color_idx]
#             rgb_color = (color.red(), color.green(), color.blue())
#
#             # Get mask for this label
#             label_mask = (self.segmentation_result == label).astype(np.uint8)
#
#             # Find contours of the segment
#             contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#             # Draw contours on the result image
#             cv2.drawContours(result, contours, -1, rgb_color, 2)
#
#         # Convert numpy array back to QImage
#         qimg = QImage(result.data, width, height, width * 3, QImage.Format_RGB888)
#         pixmap = QPixmap.fromImage(qimg)
#
#         # Scale to fit in result label
#         label_size = self.result_label.size()
#         pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#
#         self.result_label.setPixmap(pixmap)
#
#     def qimage_to_numpy(self, qimage):
#         """Convert QImage to numpy array"""
#         width = qimage.width()
#         height = qimage.height()
#
#         # Convert QImage to a format we can work with
#         if qimage.format() != QImage.Format_RGB888:
#             qimage = qimage.convertToFormat(QImage.Format_RGB888)
#
#         # Use numpy to create an array from the underlying bytes
#         bits = qimage.constBits()
#         # Create a numpy array using the buffer protocol
#         arr = np.asarray(bits).reshape(height, width, 3)
#         return arr.copy()
#

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = RegionGrowingApp()
#     window.show()
#     sys.exit(app.exec())