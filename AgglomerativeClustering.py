import numpy as np
import heapq


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
        self.centroid = centroid
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
            (self.centroid[0] * self.num_points + other.centroid[0] * other.num_points) / (self.num_points + other.num_points),
            (self.centroid[1] * self.num_points + other.centroid[1] * other.num_points) / (self.num_points + other.num_points),
            (self.centroid[2] * self.num_points + other.centroid[2] * other.num_points) / (self.num_points + other.num_points),
            (self.centroid[3] * self.num_points + other.centroid[3] * other.num_points) / (self.num_points + other.num_points),
            (self.centroid[4] * self.num_points + other.centroid[4] * other.num_points) / (self.num_points + other.num_points),
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
        spatial_distance = np.sqrt((self.centroid[0] - other.centroid[0]) ** 2 + (self.centroid[1] - other.centroid[1]) ** 2)
        color_distance = np.sqrt((self.centroid[2] - other.centroid[2]) ** 2 + (self.centroid[3] - other.centroid[3]) ** 2 + (self.centroid[4] - other.centroid[4]) ** 2)
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
            return [self.centroid[0:2]]  # return the centroid coordinates
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
        self.color_weight = color_weight
        self.spatial_weight = spatial_weight
        self.clusters = []
        self.heap = []
        self.__initialize_clusters()

    def agglomerative_clustering(self, num_clusters):
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


if __name__ == "__main__":
    # Example usage

    import cv2
    import matplotlib.pyplot as plt

    # Load the image
    image = cv2.imread('segment_images/cow-toy.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(image, color_weight=1.0, spatial_weight=1.0)
    clusters = clustering.agglomerative_clustering(num_clusters=5)

    cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]

    # Create a new image to visualize the clusters
    clustered_image = np.zeros_like(image)
    for cluster, color in zip(clusters, cluster_colors):
        pixels = cluster.get_pixels()
        for pixel in pixels:
            clustered_image[int(pixel[1]), int(pixel[0])] = color

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
