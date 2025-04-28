## seed_widget.py
import cv2
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage
import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QFileDialog, QSlider, QDialog)
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


def kmeans(image, k=3, max_iters=100):
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


def kmeans_result_to_qimage(segmented_image):
    """Convert the K-means segmentation result to a QImage"""
    # Make sure the data is in the correct format for QImage
    if segmented_image.dtype != np.uint8:
        # Scale values to 0-255 range
        if segmented_image.max() > 1.0:
            # Already in a higher range
            segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)
        else:
            # Scale from [0,1] to [0,255]
            segmented_image = (segmented_image * 255).astype(np.uint8)

    # Ensure memory is contiguous
    segmented_image = np.ascontiguousarray(segmented_image)

    # Convert to QImage
    height, width = segmented_image.shape[:2]

    if len(segmented_image.shape) == 3:
        # RGB image
        bytes_per_line = 3 * width
        return QImage(segmented_image.data, width, height,
                      bytes_per_line, QImage.Format_RGB888).copy()
    else:
        # Grayscale image
        return QImage(segmented_image.data, width, height,
                      width, QImage.Format_Grayscale8).copy()

def qimage_to_numpy(qimage):
    """Convert QImage to numpy array safely with explicit data copying"""
    # Convert to RGB888 format if it's not already
    if qimage.format() != QImage.Format_RGB888:
        qimage = qimage.convertToFormat(QImage.Format_RGB888)

    # Get dimensions
    width = qimage.width()
    height = qimage.height()

    # Create a buffer with the image data
    buffer = qimage.constBits()

    # Create numpy array using buffer protocol with explicit copying
    try:
        # For newer PyQt versions
        buffer_size = qimage.sizeInBytes()
        arr = np.frombuffer(buffer.asarray(buffer_size), dtype=np.uint8).reshape(height, width, 3)
    except AttributeError:
        # For older PyQt versions
        ptr = qimage.constBits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, width, 3)

    # Return a copy to ensure the data is not tied to the QImage's lifetime
    return arr.copy()

def cv2_to_qimage(cv_image):
    """Convert OpenCV image to QImage with proper formatting"""
    height, width = cv_image.shape[:2]

    # Handle different image types
    if len(cv_image.shape) == 2:  # Grayscale
        # For segmentation masks, ensure they are properly scaled to 0-255
        if cv_image.dtype != np.uint8:
            # Scale values to 0-255 range if needed
            cv_image = ((cv_image / cv_image.max()) * 255).astype(np.uint8)

        # Create QImage from grayscale data
        qimg = QImage(cv_image.data, width, height, width, QImage.Format_Grayscale8)
        return QImage(qimg)  # Create a copy to ensure data ownership

    elif len(cv_image.shape) == 3:  # RGB/BGR
        # Convert BGR to RGB if needed (OpenCV uses BGR by default)
        if cv_image.dtype != np.uint8:
            cv_image = cv_image.astype(np.uint8)

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * width
        qimg = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QImage(qimg)  # Create a copy to ensure data ownership

    return None

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
        image = qimage_to_numpy(image)

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
    # Signal to notify when seeds have changed
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

        # Store seeds as (x, y) tuples - removed cluster_id
        self.seeds = []
        # Colors for seeds visualization
        self.color = QColor(255, 0, 0)  # Use single color since we don't have clusters

        # State variables
        self.hovering_point = None
        self.dragging_seed = None

        self.threshold = 70  # Default threshold for region growing

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
        """Draw all seed points"""
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(self.color)

        for seed_x, seed_y in self.seeds:
            # Convert image coordinates to widget coordinates
            x = offset_x + seed_x * self.scale_factor
            y = offset_y + seed_y * self.scale_factor

            # Draw seed marker
            painter.drawEllipse(QPointF(x, y), 8, 8)

    def mousePressEvent(self, event):
        if not self.display_image:
            return

        # Check if clicking on an existing seed
        for i, (seed_x, seed_y) in enumerate(self.seeds):
            # Convert to widget coordinates
            x = (self.width() - self.display_image.width() * self.scale_factor) // 2 + seed_x * self.scale_factor
            y = (self.height() - self.display_image.height() * self.scale_factor) // 2 + seed_y * self.scale_factor

            # Check if click is within seed marker
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

            image_x = (event.pos().x() - offset_x) / self.scale_factor
            image_y = (event.pos().y() - offset_y) / self.scale_factor

            # Check if coordinates are within image bounds
            if (0 <= image_x < self.display_image.width() and
                    0 <= image_y < self.display_image.height()):
                # Add seed point as (x, y) only
                self.seeds.append((int(image_x), int(image_y)))
                self.seedsChanged.emit(self.seeds)
                self.update()

    def mouseMoveEvent(self, event):
        if not self.display_image:
            return

        # Update status bar with position
        offset_x = (self.width() - self.display_image.width() * self.scale_factor) // 2
        offset_y = (self.height() - self.display_image.height() * self.scale_factor) // 2

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
                # Update seed position (x, y only)
                self.seeds[self.dragging_seed] = (int(image_x), int(image_y))
                self.seedsChanged.emit(self.seeds)

        self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_seed = None

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.1

        # Zoom in/out based on wheel direction
        if event.angleDelta().y() > 0:
            self.scale_factor = min(self.scale_factor * zoom_factor, self.max_scale)
        else:
            self.scale_factor = max(self.scale_factor / zoom_factor, self.min_scale)

        self.update()

    def get_seed_pixels(self):
        """Return seed pixels in a format suitable for algorithms"""
        # Seeds are already in (x, y) format, so just return them
        return self.seeds


class RegionGrowingDialog(QDialog):
    # Signal to return the segmentation result to the main window
    segmentationCompleted = pyqtSignal(QImage)

    def __init__(self, parent=None, image=None):
        super().__init__(parent)
        self.setWindowTitle("Region Growing Segmentation")
        self.resize(800, 600)

        # Create main layout
        main_layout = QVBoxLayout(self)

        self.cv_image = image

        # Create seed widget
        self.seed_widget = SeedPixelWidget(self)
        self.seed_widget.seedsChanged.connect(self.handle_seeds_changed)

        # Set the image if provided
        if image is not None:
            self.seed_widget.set_image(image)

        # Create button layout
        button_layout = QHBoxLayout()

        # Create buttons
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        self.apply_button = QPushButton("Apply Segmentation")
        self.apply_button.clicked.connect(self.run_segmentation)

        # Add buttons to layout
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.apply_button)

        # Add widgets to main layout
        main_layout.addWidget(self.seed_widget)
        main_layout.addLayout(button_layout)

        # Store segmentation result
        self.segmentation_result = None

    def handle_seeds_changed(self, seeds):
        """Handle changes in seed points"""
        # Reset result when seeds change
        self.segmentation_result = None

    def run_segmentation(self):
        """Run the segmentation and emit the result"""
        try:
            seed_pixels = self.seed_widget.get_seed_pixels()
            if not seed_pixels:
                print('No seeds defined')
                return

            threshold = self.seed_widget.threshold

            # Make sure we have a valid OpenCV image
            if self.cv_image is None and self.seed_widget.image:
                # Convert QImage to OpenCV format if necessary
                self.cv_image = qimage_to_numpy(self.seed_widget.image)

            if self.cv_image is not None:
                # Run region growing
                mask = region_growing(self.cv_image, seed_pixels, threshold=threshold, use_luv=False)

                # Make sure mask is properly formatted for display (0-255 uint8)
                if mask.dtype != np.uint8:
                    # Scale to full 8-bit range
                    mask = (mask * 255 / mask.max()).astype(np.uint8)

                # Create a colored visualization of the segmentation
                height, width = mask.shape
                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

                # Color each segment with a different color
                unique_regions = np.unique(mask)
                for i, region_id in enumerate(unique_regions):
                    if region_id == 0:  # Skip background
                        continue

                    # Use a different color for each region
                    color = np.array([
                        (i * 50) % 255,  # R
                        (i * 80 + 50) % 255,  # G
                        (i * 110 + 100) % 255  # B
                    ], dtype=np.uint8)

                    # Apply color to this region
                    colored_mask[mask == region_id] = color

                # Convert colored NumPy array to QImage
                bytes_per_line = colored_mask.shape[1] * 3
                q_result = QImage(
                    colored_mask.data.tobytes(),
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888
                )

                # Important: Create a deep copy to preserve the data
                q_result = q_result.copy()

                # Store and emit result
                self.segmentation_result = q_result
                self.segmentationCompleted.emit(q_result)
                self.accept()

        except Exception as e:
            import traceback
            print(f"Error in region growing: {e}")
            traceback.print_exc()
