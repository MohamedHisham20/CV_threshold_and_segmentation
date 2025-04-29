import numpy as np
import heapq
import cv2
from PyQt5.QtGui import QImage


class Cluster:
    count = 0

    def __init__(self, centroid, left_cluster=None, right_cluster=None, neighbors=None):
        """
        Class to represent a cluster in the agglomerative clustering algorithm.
        :param centroid: (x, y, red, green, blue)
        :param left_cluster:
        :param right_cluster:
        :param neighbors: list of 8-point neighboring clusters
        """
        self.centroid = np.array(centroid, dtype=np.float64)  # Use float64 to avoid overflow
        self.left_cluster = left_cluster
        self.right_cluster = right_cluster
        self.neighbors = set(neighbors or [])
        self.merged = False
        self.num_points = 1 if left_cluster is None and right_cluster is None else 0
        if left_cluster is not None:
            self.num_points += left_cluster.num_points
            left_cluster.merged = True
        if right_cluster is not None:
            self.num_points += right_cluster.num_points
            right_cluster.merged = True

        self.id = Cluster.count
        Cluster.count += 1

    def merge(self, other):
        """
        Merge two clusters.
        :param other: other cluster to merge with
        :return: new cluster
        """
        new_centroid = (
            (self.centroid[0] * self.num_points + other.centroid[0] * other.num_points) / (
                        self.num_points + other.num_points),
            (self.centroid[1] * self.num_points + other.centroid[1] * other.num_points) / (
                        self.num_points + other.num_points),
            (self.centroid[2] * self.num_points + other.centroid[2] * other.num_points) / (
                        self.num_points + other.num_points),
            (self.centroid[3] * self.num_points + other.centroid[3] * other.num_points) / (
                        self.num_points + other.num_points),
            (self.centroid[4] * self.num_points + other.centroid[4] * other.num_points) / (
                        self.num_points + other.num_points),
        )
        new_neighbors = set(self.neighbors).union(other.neighbors)
        new_neighbors.discard(self)
        new_neighbors.discard(other)
        return Cluster(new_centroid, self, other, new_neighbors)

    def distance(self, other, color_weight=1.0, spatial_weight=1.0):
        """
        Calculate the distance between two clusters.
        :param spatial_weight:
        :param color_weight:
        :param other: other cluster to calculate distance to
        :return: distance
        """
        # Use numpy's safe distance calculation to avoid overflow
        spatial_distance = np.sqrt(np.sum(np.square(self.centroid[:2] - other.centroid[:2])))
        color_distance = np.sqrt(np.sum(np.square(self.centroid[2:] - other.centroid[2:])))

        return color_weight * color_distance + spatial_weight * spatial_distance

    def __lt__(self, other):
        # prioritize larger clusters in the heap
        return self.num_points > other.num_points

    def get_pixels(self):
        """
        Get the pixels in the cluster.
        :return: list of pixels
        """
        pixels = []
        if self.left_cluster is not None:
            pixels += self.left_cluster.get_pixels()
        if self.right_cluster is not None:
            pixels += self.right_cluster.get_pixels()
        if not pixels:
            return [(int(self.centroid[0]), int(self.centroid[1]))]  # return the centroid coordinates as integers
        return pixels


class AgglomerativeClustering:
    def __init__(self, image, color_weight=1.0, spatial_weight=1.0):
        """
        Class to perform agglomerative clustering on an image.
        :param image: input image
        :param color_weight: weight for color distance
        :param spatial_weight: weight for spatial distance
        """
        self.image = image
        # Ensure image is float to avoid overflow
        if self.image.dtype != np.float64:
            self.image = self.image.astype(np.float64)

        self.color_weight = color_weight
        self.spatial_weight = spatial_weight
        self.clusters = []
        self.heap = []
        self.__initialize_clusters()

    def agglomerate_clustering(self, num_clusters):
        num_iterations = 0
        while len(self.clusters) > num_clusters and self.heap:
            _, cluster1, cluster2 = heapq.heappop(self.heap)
            if cluster1.merged or cluster2.merged:
                continue

            if num_iterations > 1000:
                num_iterations = 0
                self.__filter_heap()

            new_cluster = cluster1.merge(cluster2)
            self.clusters.remove(cluster1)
            self.clusters.remove(cluster2)
            self.clusters.append(new_cluster)
            num_iterations += 1

            # Update neighbors
            for neighbor in new_cluster.neighbors:
                if neighbor.merged:
                    continue
                distance = new_cluster.distance(neighbor, self.color_weight, self.spatial_weight)
                heapq.heappush(self.heap, (distance, new_cluster, neighbor))

        # filter out merged clusters
        self.clusters = [cluster for cluster in self.clusters if not cluster.merged]
        return self.clusters

    def __initialize_clusters(self):
        """
        Initialize clusters from the image. Automatically builds the neighbours list for each pixel, and populates the heap
        """
        height, width, _ = self.image.shape
        for y in range(height):
            for x in range(width):
                color = self.image[y, x]
                cluster = Cluster((x, y, color[0], color[1], color[2]))
                self.clusters.append(cluster)

                # Add neighbors to the cluster
                directions = [(-1, -1), (-1, 0), (0, -1), (1, -1)]
                for dx, dy in directions:
                    if 0 > x + dx or 0 > y + dy:
                        continue

                    cluster_index = (y + dy) * width + (x + dx)
                    if cluster_index < len(self.clusters):
                        neighbor = self.clusters[cluster_index]
                        cluster.neighbors.add(neighbor)
                        neighbor.neighbors.add(cluster)

                        distance = cluster.distance(neighbor, self.color_weight, self.spatial_weight)
                        heapq.heappush(self.heap, (distance, cluster, neighbor))

    def __filter_heap(self):
        """
        Filter the heap to remove merged clusters
        """
        new_heap = []
        for _, cluster1, cluster2 in self.heap:
            if not (cluster1.merged or cluster2.merged):
                new_heap.append((_, cluster1, cluster2))
        self.heap = new_heap
        heapq.heapify(self.heap)


def agglomerate_clusters(image, num_clusters=5, color_weight=1.0, spatial_weight=1.0):
    """
    Perform agglomerative clustering on an image.
    :param image: input np image rgb  not bgr
    :param num_clusters: number of clusters
    :param color_weight: weight for color distance
    :param spatial_weight: weight for spatial distance
    :return: list of clusters
    """
    # Create a copy of the input image to avoid modifying the original
    image_copy = image.copy()

    # Convert to float64 to avoid overflows
    image_copy = image_copy.astype(np.float64)

    print("Performing agglomerative clustering...")
    clustering = AgglomerativeClustering(image_copy, color_weight, spatial_weight)
    print("Clustering initialized")
    clusters = clustering.agglomerate_clustering(num_clusters)
    print(f"Clustering completed with {len(clusters)} clusters")

    cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]

    # Add more colors if needed
    while len(cluster_colors) < len(clusters):
        cluster_colors.append([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)])

    # Create a new image to visualize the clusters
    height, width = image.shape[:2]
    clustered_image = np.zeros((height, width, 3), dtype=np.uint8)

    print("Creating clustered image...")
    for i, cluster in enumerate(clusters):
        color = cluster_colors[i % len(cluster_colors)]
        pixels = cluster.get_pixels()
        for pixel in pixels:
            # Ensure pixel coordinates are integers and within bounds
            px, py = int(pixel[0]), int(pixel[1])
            if 0 <= px < width and 0 <= py < height:
                clustered_image[py, px] = color

    print(f"Clustered image created with shape: {clustered_image.shape}")

    # Ensure the output is uint8 for proper QImage conversion
    return clustered_image.astype(np.uint8)


def cv2_to_qimage_agglomerate(cv_img):
    """
    Convert an OpenCV image to QImage with proper handling of color format
    """
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width

    # Ensure the image is in RGB format for QImage (not BGR)
    if len(cv_img.shape) == 3:  # Color image
        # OpenCV stores images in BGR order, while QImage expects RGB
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) if channel == 3 else cv_img
        return QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    else:  # Grayscale image
        return QImage(cv_img.data, width, height, width, QImage.Format_Grayscale8)


if __name__ == "__main__":
    # Example usage
    import cv2
    import matplotlib.pyplot as plt

    # Load the image
    image = cv2.imread('segment_images/cow-toy.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform agglomerative clustering
    clustered_image = agglomerate_clusters(image, num_clusters=5, color_weight=1.0, spatial_weight=1.0)

    # Display the original and clustered images
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(clustered_image)
    plt.title('Clustered Image')
    plt.axis('off')
    plt.show()